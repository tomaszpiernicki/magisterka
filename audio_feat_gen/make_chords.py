import itertools
import random
import pandas as pd
#from librosa import output
import librosa
import numpy as np

from utils import pop_many
from utils import match_lists_by_len
from sounds import RandomChord, SimpleSound


def reverse(dct):
    for key in dct.keys():
        dct[key].reverse()


def shuffle(dct):
    for key in dct.keys():
        dct[key].shuffle()


def append_fragment_to_meta(meta_list, chord, outpath, current_time):
    for sound in chord.simple_sounds:
        append_to_meta(meta_list, type(chord), outpath, current_time, (sound.onset, sound.offset), sound.midi)
        # meta = {"path": outpath, "type": type(chord), "onset": current_time + sound.onset,
        #         "offset": current_time + sound.offset,
        #         "class": sound.midi}
        # meta_list.append(meta)
    return meta_list

def append_to_meta(meta_list, neme, outpath, current_time, relative_time, midi):
    meta = {"path": outpath, "type": neme, "onset": current_time + relative_time[0],
            "offset": current_time + relative_time[1], "class": midi}
    meta_list.append(meta)
    return meta_list


def generate_single_audio(midi_key, interval, sr,  packed):
    path = packed[midi_key]
    simple_sound = SimpleSound(path, midi=midi_key, sr=sr)


def generate_random_audio_track(out_path, max_len_smpl, midi_range, overlap_list, overlap_prob, packed, sr=16000,
                                zero_padding=0):
    min_midi = min(midi_range)
    max_midi = max(midi_range)
    n_midi = max_midi - min_midi
    midi_list = []

    sound = np.zeros(zero_padding)
    meta_dict = []
    idx = 0

    # Loop generating audio and meta
    while len(sound) <= max_len_smpl:
        if not midi_list:
            midi_list = random.sample(range(min_midi, max_midi), n_midi)
        fragment_overlap = random.choices(overlap_list, weights=overlap_prob, k=1)[0]
        fragment_midi = pop_many(midi_list, fragment_overlap)
        fragment = RandomChord(idx, fragment_midi, packed, sr=sr)
        fragment_sound = fragment.generate_audio()

        current_time = librosa.samples_to_time(len(sound), sr=sr)
        sound = np.concatenate((sound, fragment_sound))
        meta_dict = append_fragment_to_meta(meta_dict, fragment, out_path, current_time)
        print(f" Appending audio at: {current_time}")
        idx += 1

    return sound, meta_dict


def generate_audio(overlap_list, overlap_prob, midi_range, packed, out_path, meta_out_path, class_labels,
                   zero_padding=None, max_len_sec=240, sr=16000):
    '''

    :param overlap_list:
    :param overlap_prob:
    :param midi_range:
    :param packed:
    :param out_path:
    :param meta_out_path:
    :param zero_padding:
    :param max_len_sec:
    :param sr:
    :return:
    '''

    if zero_padding is None:
        zero_padding = [0]
    max_len_smpl = max_len_sec * sr

    sounds = []
    meta_dicts = []
    for padding in zero_padding:
        sound_track, meta_dict_track = generate_random_audio_track(out_path, max_len_smpl, midi_range, overlap_list,
                                                                   overlap_prob, packed, sr=16000, zero_padding=padding)
        sounds.append(sound_track)
        meta_dicts.append(meta_dict_track)

    sounds = match_lists_by_len(sounds)
    sound = [sum(y) for y in zip(*sounds)]

    sound = librosa.util.normalize(sound)
    librosa.output.write_wav(out_path, sound, sr)
    print(f"Writing wave file to {out_path}")

    meta_dict = list(itertools.chain(*meta_dicts))

    df = pd.DataFrame(meta_dict)
    df.to_csv(meta_out_path, sep='\t', index=False, header=False)
    print(f"Writing meta file to {meta_out_path}")


def make_chords(folds, audio_directory, audio_filename, meta_directory, meta_filename, packed, overlap_list, overlap_prob, midi_range, class_labels, zero_padding, max_real_time, sr):
    for fold in range(folds):
        print(f"generating fold nr. {fold}")
        out_path = f"{audio_directory}/{audio_filename}_{fold}.wav"
        meta_out_path = f"{meta_directory}/{meta_filename}_{fold}.csv"
        if fold % 20 == 0:
            shuffle(packed)
            print("Shuffling packed keys")

        generate_audio(overlap_list, overlap_prob, midi_range, packed,
                       out_path, meta_out_path, class_labels, zero_padding=zero_padding,
                       max_len_sec=max_real_time, sr=sr)


#END
