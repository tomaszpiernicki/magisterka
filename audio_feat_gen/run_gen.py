import time

#import sys
#sys.path.insert(0, "..")

from run_feature_extraction import run_feature_extraction
from run_generation_dict import run_generation_dict
from run_make_chords import run_make_chords


def run_gen(config):
    run_make_chords(config)
    run_feature_extraction(config)


def run_gen_using_dict(config):
    run_generation_dict(config)
    run_feature_extraction(config)


if __name__ == '__main__':
    t = time.time()

    # run_gen("E:\Dataset\\transcription-crnn\configs\generating\\test_many_mix_notes.json")
    # run_gen_using_dict("E:/Dataset/magisterka/configs/generating/double_notes_0.5.json")
    run_gen_using_dict("E:\\Dataset\\magisterka\\configs\\generating\\valid_single_notes_full_spectr.json")
    # run_gen_using_dict("/home/piernik/magisterka/configs/generating/double_notes_0.5.json")

    elapsed = time.time() - t
    print(elapsed)
