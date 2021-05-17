import tensorflow as tf
import numpy as np
import musdb
import mir_eval
import librosa
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
musdb_folder = "../Dataset/musdb18"
mus_test = musdb.DB(root=os.path.join(dir_path, musdb_folder), subsets='test')

# Loading autoencoder model
model = tf.keras.models.load_model('Train Checkpoints/train.h5')

def estimate_and_evaluate(track):
  # Converting to mono, resampling and obtaining STFT
  channel = librosa.to_mono(track.audio.T)
  track_resampled = librosa.resample(channel, orig_sr=44100,target_sr=22050) #Resample to 22050 Hz
  mixture_ft_magn = np.abs(librosa.stft(track_resampled, n_fft=4096, hop_length=256, win_length=1024, window='hann'))
  mixture_ft_phase = np.angle(librosa.stft(track_resampled, n_fft=4096, hop_length=256, win_length=1024, window='hann'))  

  # Parameters
  freq_bins = mixture_ft_magn.shape[0]  # 2049
  time_bins = mixture_ft_magn.shape[1]
  num_frames = 9  # Each input for neural net is ~104.49 msec
  num_ft_bins = freq_bins * num_frames  # 2049(0~11.025kHz) * 9 (~104.49 msec)
  medium_frame = int(np.floor(num_frames / 2))  # 4  
  
  # Padding for frames overlapping
  pad_min = np.zeros((freq_bins, medium_frame))
  pad_min[:, :] = min(mixture_ft_magn.min(0))
  mixture_padded = np.concatenate((pad_min, mixture_ft_magn, pad_min), axis=1)  

  # Creating neural net's input
  input = np.zeros((time_bins, num_ft_bins))
  for i in range(time_bins):
      start_index = i
      end_index = start_index + num_frames
      input[i:i + 1, :] = np.transpose(
          np.reshape(mixture_padded[:, start_index:end_index], (num_ft_bins, 1), order="F"))

  # Estimating soft masks
  vocals_soft_mask = np.zeros((freq_bins, time_bins))
  for i in range(time_bins):
      predictions = model.predict(input) # Neural net's output
      pred = tf.math.sigmoid(predictions) # Apply sigmoid because last layer's activation was 'None'
      predictions = pred.numpy() # Casting to numpy array
      temp_soft_mask = np.transpose(np.reshape(model.predict(input[i:i + 1, :]), (num_frames, freq_bins)))
      vocals_soft_mask[:, i] = temp_soft_mask[:, medium_frame]
  accompaniment_soft_mask = 1 - vocals_soft_mask

  # Applying thresholds to make signals cleaner
  vocals_soft_mask[np.where(vocals_soft_mask < 0.25)]=0
  accompaniment_soft_mask[np.where(accompaniment_soft_mask < 0.75)]=0
  voc_ft_magn = np.multiply(vocals_soft_mask, mixture_ft_magn)
  acc_ft_magn = np.multiply(accompaniment_soft_mask, mixture_ft_magn)

  # Computing complex signals
  voc_complex_signal = np.multiply(voc_ft_magn, mixture_ft_phase)
  acc_complex_signal = np.multiply(acc_ft_magn, mixture_ft_phase)

  # iSTFT reconstruction of time domain signals and resampling to 44.1 kHz
  est_vocals_audio = librosa.istft(voc_complex_signal, hop_length=256, win_length=1024, window='hann', length = track_resampled.shape[0])
  est_vocals_audio = librosa.core.resample(est_vocals_audio, orig_sr=22050, target_sr=44100)
  est_accompaniment_audio = librosa.istft(acc_complex_signal, hop_length=256, win_length=1024, window='hann', length = track_resampled.shape[0])
  est_accompaniment_audio = librosa.core.resample(est_accompaniment_audio, orig_sr=22050, target_sr=44100)

  # Ground truth vocals STFT, iSTFT and resampling
  vocals_channel = librosa.to_mono(track.targets['vocals'].audio.T)
  vocals_track_resampled = librosa.resample(vocals_channel, orig_sr=44100,target_sr=22050) 
  vocals_ft_magn_gt = np.abs(librosa.stft(vocals_track_resampled, n_fft=4096, hop_length=256, win_length=1024, window='hann'))
  gt_vocals = np.multiply(vocals_ft_magn_gt, mixture_ft_phase)
  gt_vocals_audio = librosa.istft(gt_vocals, hop_length=256, win_length=1024, window='hann', length=track_resampled.shape[0])
  gt_vocals_audio = librosa.resample(gt_vocals_audio, orig_sr=22050, target_sr=44100)

  # Ground truth accompaniment STFT, iSTFT and resampling
  acc_channel = librosa.to_mono(track.targets['accompaniment'].audio.T)
  acc_track_resampled = librosa.resample(acc_channel, orig_sr=44100,target_sr=22050) 
  acc_ft_magn_gt = np.abs(librosa.stft(acc_track_resampled, n_fft=4096, hop_length=256, win_length=1024, window='hann'))
  gt_acc = np.multiply(acc_ft_magn_gt, mixture_ft_phase)
  gt_acc_audio = librosa.istft(gt_acc, hop_length=256, win_length=1024, window='hann', length=track_resampled.shape[0])
  gt_acc_audio = librosa.resample(gt_acc_audio, orig_sr=22050, target_sr=44100)

  estimates = {
      'gt_vocals':gt_vocals_audio,
      'gt_accompaniment':gt_acc_audio,
      'vocals': est_vocals_audio,
      'accompaniment': est_accompaniment_audio
  }

  return estimates

def _any_source_silent(sources):
  """Returns true if the parameter sources has any silent first dimensions"""
  return np.any(np.all(np.sum(
      sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))

# Evaluating on test set
sdr_list = []
sir_list = []
sar_list = []
for track in mus_test:
  track.chunk_start = 0.0
  track.chunk_duration = 30.0
  print('Processing... ',track.name)
  print('\n')
  while(track.chunk_start + track.chunk_duration <= track.duration):
    estimates = estimate_and_evaluate(track)
    ground_truth = np.array([]).reshape((-1,estimates['gt_vocals'].shape[0]))
    ground_truth = np.append(ground_truth,np.expand_dims(estimates['gt_vocals'], axis=0), axis=0)
    ground_truth = np.append(ground_truth,np.expand_dims(estimates['gt_accompaniment'], axis=0), axis=0)
    if(_any_source_silent(ground_truth)):
      print('At least one ground truth source is silent. Skipping current segment.')
      track.chunk_start += 30.0
      continue 
    est_sources = np.array([]).reshape((-1,estimates['vocals'].shape[0]))
    est_sources = np.append(est_sources,np.expand_dims(estimates['vocals'], axis=0), axis=0)
    est_sources = np.append(est_sources,np.expand_dims(estimates['accompaniment'], axis=0), axis=0)
    if(_any_source_silent(est_sources)):
      print('At least one estimated source is silent. Skipping current segment.')
      track.chunk_start += 30.0
      continue 
    ground_truth = ground_truth[:,0:est_sources.shape[1]]
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(ground_truth, est_sources, compute_permutation=False)
    print('segment', [track.chunk_start, track.chunk_start + track.chunk_duration])
    print('\n')
    print('metrics: vocals, accompaniment')
    print('sdr',sdr)
    print('sir',sir)
    print('sar',sar)
    print('\n')
    sdr_list.append(sdr)
    sir_list.append(sir)
    sar_list.append(sar)
    track.chunk_start += 30.0

# Printing mean values
print('mean metrics: vocals, accompaniment')
print('mean sdr',np.mean(sdr_list, axis=0))
print('mean sir',np.mean(sir_list, axis=0))
print('mean sar',np.mean(sar_list, axis=0))

# Printing standard deviation
print('std metrics: vocals, bass, drums, other')
print('std sdr',np.std(sdr_list, axis=0))
print('std sir',np.std(sir_list, axis=0))
print('std sar',np.std(sar_list, axis=0))