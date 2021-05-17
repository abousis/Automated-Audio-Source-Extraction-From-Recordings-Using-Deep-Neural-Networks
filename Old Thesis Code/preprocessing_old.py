import numpy as np
import musdb
import h5py
import librosa
import os

# Using musdb dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
musdb_folder = "../Dataset/musdb18"
mus_train = musdb.DB(root=os.path.join(dir_path, musdb_folder), subsets='train')
mus_test = musdb.DB(root=os.path.join(dir_path, musdb_folder), subsets='test')

# Parameters
num_ft_bins = 18441                                # 2049(0~11.025kHz) * 9 (~104.49 msec)
num_frames = 9                                     # Each input for CNN is ~104.49 msec
medium_frame = int(np.floor(num_frames/2))         # 4
hop_num_frames = 8  
freq_bins = 2049
chunk_per_track = 323

# Preallocation
pad_zero = np.float32(np.zeros((freq_bins, medium_frame))) # Zero padding for IBM
pad_min = np.float32(np.zeros((freq_bins, medium_frame)))  # Min padding for STFT
mixture_train_data = np.zeros((num_ft_bins, chunk_per_track))
vocals_train_data = np.zeros((num_ft_bins, chunk_per_track))
mixture_test_data = np.zeros((num_ft_bins, chunk_per_track))
vocals_test_data = np.zeros((num_ft_bins, chunk_per_track))
ibm_train_label = np.zeros((num_ft_bins, chunk_per_track))
ibm_test_label = np.zeros((num_ft_bins, chunk_per_track))

# Total frames for training and evaluation
train_frames = 94*323*4
test_frames = 50*323*4

# h5py static dataset
dataset = h5py.File(os.path.join(dir_path,'mono_dataset_shuffled_ove15.hdf5'),'w')

# Training data
train = dataset.create_dataset("train",(train_frames, num_ft_bins), chunks=(1, num_ft_bins), dtype='f', compression="gzip", compression_opts=9)
train_label = dataset.create_dataset("train_label",(train_frames,num_ft_bins), chunks=(1, num_ft_bins), dtype='f', compression="gzip", compression_opts=9)
train_auto_encoder = dataset.create_dataset("train_auto_encoder",(train_frames, num_ft_bins), chunks=(1, num_ft_bins), dtype='f', compression="gzip", compression_opts=9)

# Validation data
test = dataset.create_dataset("test",(test_frames,num_ft_bins), chunks=(1, num_ft_bins), dtype='f', compression="gzip", compression_opts=9)
test_label = dataset.create_dataset("test_label",(test_frames, num_ft_bins), chunks=(1, num_ft_bins), dtype='f', compression="gzip", compression_opts=9)
test_auto_encoder = dataset.create_dataset("test_auto_encoder",(test_frames, num_ft_bins), chunks=(1, num_ft_bins), dtype='f', compression="gzip", compression_opts=9)

train_index = 0
counter = 1

for track in mus_train:
  # Skipping tracks with < 30.0 sec duration
  if(track.duration < 30.0):
    continue
  
  # Include songs > 30 & < 60
  if(track.duration < 60.0):
    track.chunk_start = 0
  else:
    track.chunk_start = 30

  track.chunk_duration = 30
  print('Processing... ',track.name)
  while(track.chunk_start + track.chunk_duration < 140.0 and track.chunk_start + track.chunk_duration <= track.duration):
    print('Clip', counter)
    print('segment', [track.chunk_start, track.chunk_start + track.chunk_duration])    
    track_resampled = librosa.resample(track.audio.T, orig_sr=44100,target_sr=22050) #Resample to 22050 Hz
    mixture_ft_magn = np.abs(librosa.stft(librosa.to_mono(track_resampled), n_fft=4096, hop_length=256, win_length=1024, window='hann')) 


    pad_min[:,:] = min(mixture_ft_magn.min(0))
    mixture_padded = np.concatenate((pad_min, mixture_ft_magn, pad_min), axis=1)
  
    # Append vocals
    vocals_resampled = librosa.resample(track.sources['vocals'].audio.T, orig_sr=44100, target_sr=22050)
    vocals_ft_magn = np.abs(librosa.stft(librosa.to_mono(vocals_resampled), n_fft=4096, hop_length=256, win_length=1024, window='hann'))
    # Create Binary Mask
    ideal_binary_mask = (vocals_ft_magn > mixture_ft_magn).astype('float32')

    # Concatenation for frame overlapping
    vocals_padded = np.concatenate((pad_min, np.multiply(vocals_ft_magn, ideal_binary_mask), pad_min), axis=1)
    ibm_padded = np.concatenate((pad_zero, ideal_binary_mask, pad_zero), axis=1)
    for i in range(chunk_per_track):
              start_index = i*hop_num_frames
              end_index = start_index + num_frames
              mixture_train_data[:, i:i+1] = np.reshape(mixture_padded[:, start_index:end_index], (num_ft_bins, 1), order="F")
              vocals_train_data[:, i:i+1] = np.reshape(vocals_padded[:, start_index:end_index], (num_ft_bins, 1), order="F")
              ibm_train_label[:, i:i+1] = np.reshape(ibm_padded[:,start_index:end_index], (num_ft_bins, 1), order="F")
    
    #Shuffling
    col_idxs = np.random.permutation(chunk_per_track)
    mixture_train_data = mixture_train_data[:, col_idxs]
    vocals_train_data = vocals_train_data[:, col_idxs]
    ibm_train_label = ibm_train_label[:, col_idxs]

    start = train_index*chunk_per_track
    end = (train_index+1)*chunk_per_track

    train[start:end,:] = np.transpose(mixture_train_data)
    train_label[start:end,:] = np.transpose(ibm_train_label)
    train_auto_encoder[start:end,:] = np.transpose(vocals_train_data)
    train_index += 1
    
    track.chunk_start += 15.0
    counter += 1

test_index = 0
counter = 1

for track in mus_test:
  # Skipping tracks with < 30.0 sec duration
  if(track.duration < 30.0):
    continue
  
  # Include songs > 30 & < 60
  if(track.duration < 60.0):
    track.chunk_start = 0
  else:
    track.chunk_start = 30

  track.chunk_duration = 30
  print('Processing... ',track.name)
  while(track.chunk_start + track.chunk_duration < 140.0 and track.chunk_start + track.chunk_duration <= track.duration):
    print('Clip', counter)
    print('segment', [track.chunk_start, track.chunk_start + track.chunk_duration])    
    track_resampled = librosa.resample(track.audio.T, orig_sr=44100,target_sr=22050) #Resample to 22050 Hz
    mixture_ft_magn = np.abs(librosa.stft(librosa.to_mono(track_resampled), n_fft=4096, hop_length=256, win_length=1024, window='hann')) 


    pad_min[:,:] = min(mixture_ft_magn.min(0))
    mixture_padded = np.concatenate((pad_min, mixture_ft_magn, pad_min), axis=1)
  
    # Append vocals
    vocals_resampled = librosa.resample(track.sources['vocals'].audio.T, orig_sr=44100, target_sr=22050)
    vocals_ft_magn = np.abs(librosa.stft(librosa.to_mono(vocals_resampled), n_fft=4096, hop_length=256, win_length=1024, window='hann'))
    # Create Binary Mask
    ideal_binary_mask = (vocals_ft_magn > mixture_ft_magn).astype('float32')

    # Concatenation for frame overlapping
    vocals_padded = np.concatenate((pad_min, np.multiply(vocals_ft_magn, ideal_binary_mask), pad_min), axis=1)
    ibm_padded = np.concatenate((pad_zero, ideal_binary_mask, pad_zero), axis=1)
    for i in range(chunk_per_track):
              start_index = i*hop_num_frames
              end_index = start_index + num_frames
              mixture_test_data[:, i:i+1] = np.reshape(mixture_padded[:, start_index:end_index], (num_ft_bins, 1), order="F")
              vocals_test_data[:, i:i+1] = np.reshape(vocals_padded[:, start_index:end_index], (num_ft_bins, 1), order="F")
              ibm_test_label[:, i:i+1] = np.reshape(ibm_padded[:, start_index:end_index], (num_ft_bins, 1), order="F")
    
    #Shuffling
    col_idxs = np.random.permutation(chunk_per_track)
    mixture_test_data = mixture_test_data[:, col_idxs]
    vocals_test_data = vocals_test_data[:, col_idxs]
    ibm_test_label = ibm_test_label[:, col_idxs]

    start = test_index*chunk_per_track
    end = (test_index+1)*chunk_per_track

    test[start:end,:] = np.transpose(mixture_test_data)
    test_label[start:end,:] = np.transpose(ibm_test_label)
    test_auto_encoder[start:end,:] = np.transpose(vocals_test_data)
    test_index += 1
    
    track.chunk_start += 15.0
    counter += 1

dataset.close()
