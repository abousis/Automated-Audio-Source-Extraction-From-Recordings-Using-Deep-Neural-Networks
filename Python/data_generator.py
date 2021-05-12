import numpy as np
import musdb
import librosa
import random
import tensorflow.keras
from time import time
import os

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, steps_per_epoch, tracks_in_batch, subsets, split):
        self.steps_per_epoch = steps_per_epoch
        self.tracks_in_batch = tracks_in_batch
        self.subsets = subsets
        self.split = split
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.musdb_folder = "../Dataset/musdb18"
        self.mus = musdb.DB(root=os.path.join(
            self.dir_path, self.musdb_folder), subsets=subsets, split=split)
        self.track_number = np.arange(len(self.mus))
        np.random.shuffle(self.track_number)
        self.cur_index = 0
        self.freq_bins = 2049
        self.num_frames = 9
        self.num_ft_bins = self.freq_bins * self.num_frames  # 18441
        self.medium_frame = int(np.floor(self.num_frames / 2))  # 4
        self.hop_num_frames = 8
        self.chunk_per_track = 323
        self.duration = 30  # 30 sec track chunks


    def __len__(self):
        return self.steps_per_epoch


    def __getitem__(self, index):
        mixes = []
        targets = []

        # Random mixing batches
        for i in range(self.tracks_in_batch):
            # Preallocation with zeros
            pad_zero = np.float32(np.zeros((self.freq_bins, self.medium_frame)))  # Zero padding for IBM
            pad_min = np.float32(np.zeros((self.freq_bins, self.medium_frame)))  # Minimum padding for frames overlapping
            mixture_train_data = np.float32(np.zeros((self.num_ft_bins, self.chunk_per_track)))
            ibm_train_label = np.float32(np.zeros((self.num_ft_bins, self.chunk_per_track)))

            mix, target_vocals = self.get_random_track_piece()
            if self.is_source_silent(mix):
                continue
            track_resampled = librosa.core.resample(mix, orig_sr=44100, target_sr=22050)  # Resample to 22050 Hz
            mixture_ft_magn = np.float32(
                np.abs(librosa.stft(track_resampled, n_fft=4096, hop_length=256, win_length=1024, window='hann')))
            pad_min[:, :] = np.float32(min(mixture_ft_magn.min(0)))
            mixture_padded = np.concatenate((pad_min, mixture_ft_magn, pad_min), axis=1)

            # Append vocals
            vocals_resampled = librosa.core.resample(target_vocals, orig_sr=44100, target_sr=22050)
            vocals_ft_magn = np.float32(
                np.abs(librosa.stft(vocals_resampled, n_fft=4096, hop_length=256, win_length=1024, window='hann')))

            # Create Binary Mask
            ideal_binary_mask = (vocals_ft_magn > mixture_ft_magn).astype('float32')

            # Concatenation for frame overlapping
            ibm_padded = np.concatenate((pad_zero, ideal_binary_mask, pad_zero), axis=1)
            for j in range(self.chunk_per_track):
                start_index = j * self.hop_num_frames
                end_index = start_index + self.num_frames
                mixture_train_data[:, j:j + 1] = np.reshape(mixture_padded[:, start_index:end_index], (self.num_ft_bins, 1),
                                                            order="F")
                ibm_train_label[:, j:j + 1] = np.reshape(ibm_padded[:, start_index:end_index], (self.num_ft_bins, 1),
                                                        order="F")

            mixes = mixes + list(np.transpose(mixture_train_data))
            targets = targets + list(np.transpose(ibm_train_label))

        mix_batch = np.array(mixes)
        target_batch = np.array(targets)
        return mix_batch, target_batch


    def get_random_track_piece(self):
        # Getting random track
        random.seed(int(time() % 1 * 6000))
        if self.cur_index == self.track_number.shape[0]:
            np.random.shuffle(self.track_number)
            self.cur_index = 0
        track = self.mus[self.track_number[self.cur_index]]
        self.cur_index += 1
        # Getting random track chunk of given duration
        track.chunk_duration = self.duration
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        mix = track.audio.T
        vocals = track.targets['vocals'].audio.T
        # Random swapping channels
        channel = random.randint(0, mix.shape[0] - 1)
        return mix[channel], vocals[channel]


    def is_source_silent(self, source):
        # Returns true if the parameter source is fully silent
        return not np.any(source)
