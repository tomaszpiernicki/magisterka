import time

from audio_feat_gen.generation_dict import generate_audio_with_dict
from audio_feat_gen.make_chords import make_chords
from configuration import GeneratorConfig


def run_generation_dict(config_file=None):
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

    generate_audio_with_dict(folds, audio_directory, audio_filename, meta_directory, meta_filename, packed, overlap_list,
                overlap_prob, midi_range, class_labels, zero_padding, max_real_time, sr, bpm = 60)

    # make_chords(folds, audio_directory, audio_filename, meta_directory, meta_filename, packed, overlap_list,
    #             overlap_prob, midi_range, class_labels, zero_padding, max_real_time, sr)

if __name__ == '__main__':
    t = time.time()

    run_generation_dict("E:\Dataset\\transcription-crnn\configs\generating\\triple_notes_0.5.json")

    elapsed = time.time() - t
    print(elapsed)
