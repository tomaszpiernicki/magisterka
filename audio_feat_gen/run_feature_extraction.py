import time

from audio_feat_gen.feature import feature_extraction
from configuration import FeatureXtractConfig


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
    run_feature_extraction("E:\\Dataset\\transcription-crnn\\configs\\generating\\valid_quint_notes.json")

    elapsed = time.time() - t
    print(elapsed)
