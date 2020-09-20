import time

from audio_feat_gen.make_chords import make_chords
from configuration import GeneratorConfig


def run_make_chords(config_file=None):
    configuration = GeneratorConfig()
    configuration.parse_config(config_file)
    configuration.save_config()

    folds = configuration.folds
    audio_directory = configuration.audio_directory
    audio_filename = configuration.audio_filename
    meta_directory = configuration.meta_directory
    meta_filename = configuration.meta_filename
    overlap_list = configuration.overlap_list
    overlap_prob = configuration.overlap_prob
    midi_range = configuration.midi_range
    class_labels = configuration.class_labels
    zero_padding = configuration.zero_padding
    max_real_time = configuration.max_real_time
    sr = configuration.sr

    packed = configuration.packed_paths

    make_chords(folds, audio_directory, audio_filename, meta_directory, meta_filename, packed, overlap_list,
                overlap_prob, midi_range, class_labels, zero_padding, max_real_time, sr)

if __name__ == '__main__':
    t = time.time()

    run_make_chords("E:\Dataset\\transcription-crnn\configs\generating\\valid_quint_notes.json")

    elapsed = time.time() - t
    print(elapsed)