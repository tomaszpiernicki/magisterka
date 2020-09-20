import time

from audio_feat_gen.run_feature_extraction import run_feature_extraction
from audio_feat_gen.run_generation_dict import run_generation_dict
from audio_feat_gen.run_make_chords import run_make_chords


def run_gen(config):
    run_make_chords(config)
    run_feature_extraction(config)


def run_gen_using_dict(config):
    run_generation_dict(config)
    run_feature_extraction(config)


if __name__ == '__main__':
    t = time.time()

    # run_gen("E:\Dataset\\transcription-crnn\configs\generating\\test_many_mix_notes.json")
    run_gen_using_dict("E:\Dataset\\transcription-crnn\configs\generating\\triple_notes_0.5.json")

    elapsed = time.time() - t
    print(elapsed)