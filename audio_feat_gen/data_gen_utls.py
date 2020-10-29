import os
import random
import re

import numpy as np

import librosa
from librosa import load

import matplotlib.pylab as plt

def plot_feature_map(feature, title):

    plt.title(title)

    xs = range(feature.shape[1])
    plt.imshow(feature, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()


def load_audio(filename, sr):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        return load(filename, sr=sr)
    else:
        raise NotImplementedError("Only .wav supported")


def apply_time_tresholding(meta_vector, audible_threshold):
    audible_threshold = meta_vector.shape[0] / audible_threshold

    audible_threshold = audible_threshold
    meta_vector[meta_vector < audible_threshold] = 0
    meta_vector[meta_vector >= audible_threshold] = 1

    return meta_vector



def extract_sub_dir(path):
    try:
        result = re.findall('-[0123456789][0123456789][0123456789]-', path)[0]
    except Exception as IndexError:
        result = ''
    return result[1:-1]


class RoundRobin():
    def __init__(self):
        self.lst = []

    def get(self):
        to_return = self.lst[0]
        self.lst = self.lst[1:]
        self.add(to_return)
        return to_return

    def add(self, path):
        self.lst.append(path)

    def reverse(self):
        self.lst.reverse()

    def shuffle(self):
        random.shuffle(self.lst)

    def remove(self, index):
        print("Not implemented yet.")

    def __len__(self):
        return len(self.lst)


def pack_paths(files):
    path_dict = {}

    for file in files:
        k = int(extract_sub_dir(file))
        if k not in path_dict:
            path_dict[k] = RoundRobin()
        path_dict[k].add(file)

    return path_dict


def save_in_parts(inputs, targets, fold_idx, fold_size, feat_folder, is_mono, audible_threshold=2):
    assert inputs.shape[0] == targets.shape[0], "Input tensor and label tensor should have same time dimention."

    reminder = inputs.shape[0] % fold_size

    inputs = inputs[0:-reminder][:]
    targets = targets[0:-reminder][:]

    full_time_len = inputs.shape[0]
    n_small_folds = int(full_time_len / fold_size)

    inputs = inputs.reshape(n_small_folds, fold_size, inputs.shape[1])
    targets = targets.reshape(n_small_folds, fold_size, targets.shape[1])

    audible_threshold = fold_size / audible_threshold

    skipped = 0
    for idx in range(n_small_folds):
        temp_input = inputs[idx]
        temp_labels = targets[idx]

        temp_labels = np.sum(temp_labels, 0)
        temp_labels[temp_labels < audible_threshold] = 0
        temp_labels[temp_labels >= audible_threshold] = 1

        # normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}_{}.npz'.format('mon' if is_mono else 'bin', fold_idx, idx))

        if np.sum(temp_labels) == 0:
            print("Skipping, empty.")
            skipped += 1
            continue

        print("Saving.")
        normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}_{}.npz'.format('mon' if is_mono else 'bin',
                                                                                       fold_idx, idx))
        np.savez(normalized_feat_file, temp_input, temp_labels)
    print(f"Saved:{n_small_folds - skipped}")
    print(f"Skipped{skipped}")



def save_in_parts_signle(inputs, fold_size, temp_folder, fold_idx, is_mono  = True):
    reminder = inputs.shape[0] % fold_size

    inputs = inputs[0:-reminder][:]

    full_time_len = inputs.shape[0]
    n_small_folds = int(full_time_len / fold_size)

    inputs = inputs.reshape(n_small_folds, fold_size, inputs.shape[1])

    skipped = 0
    for idx in range(n_small_folds):
        temp_input = inputs[idx]

        # normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}_{}.npz'.format('mon' if is_mono else 'bin', fold_idx, idx))


        print("Saving.")
        normalized_feat_file = os.path.join(f"{temp_folder}/features", 'mbe_{}_fold{}_{}.npz'.format('mon' if is_mono else 'bin',
                                                                                       fold_idx, idx))
        np.savez(normalized_feat_file, temp_input)
    print(f"Saved:{n_small_folds - skipped}")
    print(f"Skipped{skipped}")


def load_desc_file(_desc_file, class_labels, map_midis = True):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        if map_midis:
            _desc_dict[name].append([float(words[2]), float(words[3]), class_labels[str(int(words[-1]))]])
        else:
            _desc_dict[name].append([float(words[2]), float(words[3]), int(words[-1])])
    return _desc_dict


def extract_mbe(_y, _sr, _nfft, _nb_mel, mode="mel"):
    _y /= np.max(np.abs(_y), axis=0)
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=int(_nfft/2), power=1)
    # plot_feature_map(np.log(spec), 'spectrogram stft')

    if mode == "spec":
        return np.log(spec)
    elif mode == "mel":
        mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
        with np.errstate(divide='ignore'):
            return np.log(np.dot(mel_basis, spec))
        # plot_feature_map(to_return, 'melspectrogram')