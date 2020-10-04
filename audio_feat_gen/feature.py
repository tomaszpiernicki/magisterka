import sys
import os
import numpy as np


from sklearn import preprocessing

from data_gen_utls import load_desc_file, load_audio, extract_mbe, save_in_parts


def feature_normalization(evaluation_setup_folder, meta_file_name, fold, class_labels, feat_folder, is_mono):
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

    x_train[x_train == float('-inf')] = sys.float_info.min  # minimal posdible value
    x_train = scaler.fit_transform(x_train)
    return x_train, y_train


def extract_features(audio_folder, is_mono, nfft, nb_mel_bands, class_labels, desc_dict, hop_len, feat_folder, sr, feat_mode):
    for audio_filename in os.listdir(audio_folder):
        audio_file = os.path.join(audio_folder, audio_filename)
        print(f'Extracting features and label for : {audio_file}, with mode {feat_mode}')
        y, sr = load_audio(audio_file, sr)
        mbe = None
        if is_mono:
            mbe = extract_mbe(y, sr, nfft, nb_mel_bands, mode=feat_mode).T
        else:
            for ch in range(y.shape[0]):
                mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands, mode=feat_mode).T
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


def feature_extraction(folds, evaluation_setup_folder, meta_file_name, audio_folder, feat_folder, sr, is_mono, nfft, hop_len, nb_mel_bands, class_labels, fold_size, audible_threshold, feat_mode):
    # -----------------------------------------------------------------------
    # Feature extraction and label generation
    # -----------------------------------------------------------------------
    desc_dict = {}
    for fold in range(folds):
        file = f"{evaluation_setup_folder}/{meta_file_name}_{fold}.csv"
        desc_dict.update(load_desc_file(file, class_labels))

    # Extract features for all audio files, and save it along with labels
    extract_features(audio_folder, is_mono, nfft, nb_mel_bands, class_labels, desc_dict, hop_len, feat_folder, sr, feat_mode)

    # Feature Normalization
    for fold in range(folds):
        x_train, y_train = feature_normalization(evaluation_setup_folder, meta_file_name, fold, class_labels, feat_folder, is_mono)
        save_in_parts(x_train, y_train, fold, fold_size, feat_folder, is_mono, audible_threshold=audible_threshold)

    # cleanup
    for fold in range(folds):
        train_file = f"{evaluation_setup_folder}/{meta_file_name}_{fold}.csv"
        train_dict = load_desc_file(train_file, class_labels)

        for key in train_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
            os.remove(tmp_feat_file)
# END
