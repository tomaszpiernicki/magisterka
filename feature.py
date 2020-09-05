import sys
import os
import time
import numpy as np

import librosa
from librosa import load

from sklearn import preprocessing

from configuration import FeatureXtractConfig

def load_audio(filename, sr):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        data, sr = load(filename, sr=sr)
        return data, sr
    return None, None

def apply_time_tresholding(meta_vector, audible_threshold):
    audible_threshold = meta_vector.shape[0] / audible_threshold

    audible_threshold = audible_threshold
    meta_vector[meta_vector < audible_threshold] = 0
    meta_vector[meta_vector >= audible_threshold] = 1

    return meta_vector

def apply_time_tresholding(meta_vector, audible_threshold):
    audible_threshold = meta_vector.shape() / audible_threshold

    audible_threshold = audible_threshold
    meta_vector[meta_vector < audible_threshold] = 0
    meta_vector[meta_vector >= audible_threshold] = 1

    return meta_vector

def save_in_parts(X_input, Y_labels, fold_idx, fold_size, feat_folder, is_mono, audible_threshold=2):
    assert X_input.shape[0] == Y_labels.shape[0], "Input tensor and label tensor should have same time dimention."

    reminder = X_input.shape[0] % fold_size

    X_input = X_input[0:-reminder][:]
    Y_labels = Y_labels[0:-reminder][:]

    full_time_len = X_input.shape[0]
    n_small_folds = int(full_time_len / fold_size)

    X_input = X_input.reshape(n_small_folds, fold_size, X_input.shape[1])
    Y_labels = Y_labels.reshape(n_small_folds, fold_size, Y_labels.shape[1])

    audible_threshold = fold_size / audible_threshold

    skipped = 0
    for idx in range(n_small_folds):
        temp_input = X_input[idx]
        temp_labels = Y_labels[idx]

        temp_labels = np.sum(temp_labels, 0)
        temp_labels[temp_labels < audible_threshold] = 0
        temp_labels[temp_labels >= audible_threshold] = 1

        normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}_{}.npz'.format('mon' if is_mono else 'bin', fold_idx, idx))

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

def load_desc_file(_desc_file, class_labels):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), class_labels[str(int(words[-1]))]])
    return _desc_dict


def extract_mbe(_y, _sr, _nfft, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=int(_nfft/2), power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    with np.errstate(divide='ignore'):
        return np.log(np.dot(mel_basis, spec))


def feature_extraction(folds, evaluation_setup_folder, meta_file_name, audio_folder, feat_folder, sr, is_mono, nfft, hop_len, nb_mel_bands, class_labels, fold_size, audible_threshold):
    # -----------------------------------------------------------------------
    # Feature extraction and label generation
    # -----------------------------------------------------------------------
    desc_dict = {}
    for fold in range(folds):
        file = train_file = f"{evaluation_setup_folder}/{meta_file_name}_{fold}.csv"
        desc_dict.update(load_desc_file(file, class_labels))

    # Extract features for all audio files, and save it along with labels
    for audio_filename in os.listdir(audio_folder):
        audio_file = os.path.join(audio_folder, audio_filename)
        print('Extracting features and label for : {}'.format(audio_file))
        y, sr = load_audio(audio_file, sr)
        mbe = None

        if is_mono:
            mbe = extract_mbe(y, sr, nfft, nb_mel_bands).T
        else:
            for ch in range(y.shape[0]):
                mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
                if mbe is None:
                    mbe = mbe_ch
                else:
                    mbe = np.concatenate((mbe, mbe_ch), 1)

        label = np.zeros((mbe.shape[0], len(class_labels)))
        tmp_data = np.array(desc_dict[audio_filename])
        frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
        frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
        se_class = tmp_data[:, 2].astype(int)
        for ind, val in enumerate(se_class):
            label[frame_start[ind]:frame_end[ind], val] = 1
        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
        np.savez(tmp_feat_file, mbe, label)

    # -----------------------------------------------------------------------
    # Feature Normalization
    # -----------------------------------------------------------------------

    for fold in range(folds):
        train_file = f"{evaluation_setup_folder}/{meta_file_name}_{fold}.csv"
        train_dict = load_desc_file(train_file, class_labels)

        x_train, y_train = None, None
        for key in train_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
            dmp = np.load(tmp_feat_file)
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
            if x_train is None:
                x_train, y_train = tmp_mbe, tmp_label
            else:
                x_train, y_train = np.concatenate((x_train, tmp_mbe), 0), np.concatenate((y_train, tmp_label), 0)

        # Normalize the training data, and scale the testing data using the training data weights
        scaler = preprocessing.StandardScaler()

        x_train[x_train == float('-inf')] = sys.float_info.min   # minimal posdible value
        x_train = scaler.fit_transform(x_train)

        save_in_parts(x_train, y_train, fold, fold_size, feat_folder, is_mono, audible_threshold=audible_threshold)

def run_feature_extraction(config_file=None):
    configuration = FeatureXtractConfig()
    configuration.parse_config(config_file)

    sr = configuration.sr
    class_labels = configuration.class_labels
    audio_folder = configuration.audio_directory
    feat_folder = configuration.feature_folder
    folds = configuration.folds
    meta_filename = configuration.meta_filename

    meta_directory = configuration.meta_directory
    is_mono = configuration.is_mono
    nfft = configuration.nfft
    hop_len = configuration.hop_len
    nb_mel_bands = configuration.nb_mel_bands
    fold_size = configuration.fold_size
    audible_threshold = configuration.audible_threshold

    feature_extraction(folds, meta_directory, meta_filename, audio_folder, feat_folder, sr, is_mono, nfft,
                       hop_len, nb_mel_bands, class_labels, fold_size, audible_threshold)

if __name__ == '__main__':
    t = time.time()
    run_feature_extraction("configs/generate_double_notes_long_many_midis.json")

    elapsed = time.time() - t
    print(elapsed)

# END
