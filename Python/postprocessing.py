import tensorflow as tf
import numpy as np
import librosa
import os
import musdb

base_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = "../Models/model_v2.h5" # Load pre-trained model
model = tf.keras.models.load_model(os.path.join(base_dir, model_dir), compile=False)

@staticmethod
def estimate_sources(audio):
    # Resampling to 22.05 kHz
    channel = librosa.core.resample(audio, orig_sr=44100, target_sr=22050)

    # Calculating Short Time Fourier Transform
    mixture_ft_magn = np.abs(librosa.stft(channel, n_fft=4096, win_length=1024, hop_length=256, window='hann'))
    mixture_ft_phase = np.angle(librosa.stft(channel, n_fft=4096, win_length=1024, hop_length=256, window='hann'))

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
        temp_soft_mask = np.transpose(np.reshape(model.predict(input[i:i + 1, :]), (num_frames, freq_bins)))
        vocals_soft_mask[:, i] = temp_soft_mask[:, medium_frame]
    accompaniment_soft_mask = 1 - vocals_soft_mask

    # Applying thresholds to make signals cleaner
    vocals_soft_mask[vocals_soft_mask < 0.2] = 0
    accompaniment_soft_mask[accompaniment_soft_mask < 0.85] = 0
    voc_ft_magn = np.multiply(vocals_soft_mask, mixture_ft_magn)
    acc_ft_magn = np.multiply(accompaniment_soft_mask, mixture_ft_magn)

    # Computing complex signals
    voc_complex_signal = np.multiply(voc_ft_magn, mixture_ft_phase)
    acc_complex_signal = np.multiply(acc_ft_magn, mixture_ft_phase)

    # iSTFT reconstruction of time domain signals and resampling to 44.1 kHz
    vocals_audio = librosa.istft(voc_complex_signal, hop_length=256, win_length=1024, window='hann')
    vocals_audio = librosa.core.resample(vocals_audio, orig_sr=22050, target_sr=44100)
    accompaniment_audio = librosa.istft(acc_complex_signal, hop_length=256, win_length=1024, window='hann')
    accompaniment_audio = librosa.core.resample(accompaniment_audio, orig_sr=22050, target_sr=44100)

    estimates = {
        'vocals': vocals_audio,
        'accompaniment': accompaniment_audio,
    }

    return estimates

#### Usage example ####

# def cut_track(track, duration):
#   track.chunk_duration = duration
#   track.chunk_start = np.random.uniform(0, track.duration - track.chunk_duration)
#   return track

# mus = musdb.DB(root=os.path.join(base_dir, "../Dataset/musdb18"), subsets="test", split="")
# track = cut_track(mus[0], 30) 
# audio = librosa.to_mono(track.audio.T)
# estimates = estimate_sources(audio)

# Write vocals
# librosa.output.write_wav(base_dir + "/vocals.wav", y=estimates['vocals'], sr=44100)
# Write accompaniment
# librosa.output.write_wav(base_dir + "/accompaniment.wav", y=estimates['accompaniment'], sr=44100)