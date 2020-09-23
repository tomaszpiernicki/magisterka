import itertools
import random
from copy import deepcopy

import librosa
import pandas as pd
#from librosa import output

from augmentations import Gain, AddNoise, PolarityInversion
from make_chords import append_to_meta, shuffle

import numpy as np

import matplotlib.pylab as plt

from utils import match_lists_by_len
from sounds import SoundFactory


def resolve_capacity(overlap_prob, overlap_list, max_samples, class_labels):
    number_classes = len(list(class_labels.keys()))

    capacities = {}
    for idx, prob in enumerate(overlap_prob):
        overlap = overlap_list[idx]
        capacities[overlap] = {}
        overlap_max = prob / 100 * max_samples
        capcity = overlap_max / number_classes * overlap_list[idx]
        for key in class_labels.keys():
            capacities[overlap][key] = round(capcity)

    return capacities


def get_keys_with_capacity(current_sample_idx, note, overlap, capacity, the_dict):

    temp_midi_rand = deepcopy(capacity)
    for o_idx in range(overlap):
        midi_key = random.choice(list(temp_midi_rand.keys()))

        max_capacity = max(capacity.values())
        # print(max_capacity, print(notes))
        if note > max_capacity:
            note = max_capacity

        loop_count = 0
        while True:
            if capacity[midi_key] >= note and midi_key in temp_midi_rand.keys():    # TODO: Inf# loops?
                the_dict[midi_key].append((current_sample_idx, current_sample_idx + note))
                capacity[midi_key] -= note
                break
            else:
                loop_count += 1
                midi_key = int(midi_key)
                min_key = int(list(the_dict.keys())[0])
                max_key = int(list(the_dict.keys())[-1])
                midi_key += 1
                if midi_key > max_key:
                    midi_key = min_key
                midi_key = str(midi_key)

        temp_midi_rand.pop(midi_key)
    return the_dict, capacity


class Notes:
    def __init__(self, sr, bpm):
        self.notes = {"quarter": sr * 60 / bpm}
        self.notes["sixth"] = self.notes["quarter"] / 4
        self.notes["eight"] = self.notes["quarter"] / 2
        self.notes["half"] = self.notes["quarter"] * 2
        self.notes["full"] = self.notes["quarter"] * 4

    def get_random(self):
        return self.notes[random.choice(list(self.notes.keys()))]

    def check_minimum(self, max_capacity):
        for key in self.notes:
            if self.notes[key] > max_capacity:
                self.notes[key] = max_capacity


def generation_dict(overlap_list, overlap_prob, sr, bpm, max_samples, midi_range):
    note_types = {}
    for overlap in overlap_list:
        note_types[overlap] = Notes(sr, bpm)

    max_midi, min_midi = max(midi_range), min(midi_range)

    gen_dict = {}
    for i in range(max_midi - min_midi + 1):
        gen_dict[f"{min_midi + i}"] = []

    overlap_prob_dict = {}
    for idx, o in enumerate(overlap_list):
        overlap_prob_dict[o] = overlap_prob[idx]

    capacities = resolve_capacity(overlap_prob, overlap_list, max_samples, gen_dict)

    current_sample_idx = 0
    while capacities.keys():
        overlap_prob_temp = [overlap_prob_dict[o] for o in list(overlap_prob_dict.keys())]
        overlap = random.choices([int(cap) for cap in list(capacities.keys())], weights=itertools.accumulate(overlap_prob_temp), k=1)[0]
        # overlap = random.choices([int(cap) for cap in list(capacities.keys())], k=1)[0]
        note_samples = note_types[overlap].get_random()
        gen_dict, capacities[overlap] = get_keys_with_capacity(current_sample_idx, note_samples, overlap, capacities[overlap], gen_dict)
        current_sample_idx = current_sample_idx + note_samples
        # overlap_max = max(capacities[overlap].values())
        overlap_max = sorted(capacities[overlap].values(), reverse=True)[overlap]
        note_types[overlap].check_minimum(overlap_max)
        if overlap_max == 0:
            capacities.pop(overlap)
            overlap_prob_dict.pop(overlap)
            overlap_prob_temp = [overlap_prob_dict[o] for o in list(overlap_prob_dict.keys())]
            overlap_prob_temp = [int(float(i) / max(overlap_prob_temp) * 100) for i in overlap_prob_temp]
            for idx, o in enumerate(overlap_prob_dict.keys()):
                overlap_prob_dict[o] = overlap_prob_temp[idx]

    return gen_dict


def analyze_gen_dict(gen_dict):
    dict_count = {}
    for key in gen_dict.keys():
        dict_count[key] = 0
        for interval in gen_dict[key]:
            dict_count[key] += interval[1] - interval[0]
    plt.bar(list(dict_count.keys()), list(dict_count.values()))
    plt.show()


def generation_dict_to_audio(packed_paths, sr, gen_dict, out_path, meta_out_path, max_samples, augmentations, fold, folds):
    sound_factory = SoundFactory(packed_paths, sr, augmentations=augmentations)

    meta_list = []
    audio_array = np.zeros(max_samples)
    for key_idx, key in enumerate(gen_dict.keys()):
        for interval_idx, interval in enumerate(gen_dict[key]):
            # print(
            #     f"Parsing key: {key}, {int(100 * key_idx / len(list(gen_dict.keys())))}%, interval: {interval}, {int(100 * interval_idx / len(gen_dict[key]))}%")
            print(f"Parsing: {fold + 1/folds * 100}%, {int(100 * (key_idx + 1) / len(list(gen_dict.keys())))}%, {int(100 * (interval_idx + 1) / len(gen_dict[key]))}%")

            y, relative_time_interval = sound_factory.get_note(key, interval)
            interval = (int(interval[0]), int(interval[1]))
            audio_array_temp = audio_array[slice(*interval)]
            (audio_array_temp, y) = match_lists_by_len([audio_array_temp, y])
            added = np.add(audio_array_temp, y)
            normal = librosa.util.normalize(added)
            try:
                audio_array[slice(*interval)] = normal
                current_time = librosa.samples_to_time(interval[0], sr=sr)
                meta_list = append_to_meta(meta_list, key, out_path, current_time, relative_time_interval, key)
            except:
                pass

    audio_array = 0.3 * audio_array
    librosa.output.write_wav(out_path, audio_array, sr)
    df = pd.DataFrame(meta_list)
    df.to_csv(meta_out_path, sep='\t', index=False, header=False)
    print(f"saving to :{out_path}")


def generate_audio_with_dict(folds, audio_directory, audio_filename, meta_directory, meta_filename, packed_paths, overlap_list,
                overlap_prob, midi_range, class_labels, zero_padding, max_real_time, sr, bpm):
    max_samples = sr * max_real_time

    for fold in range(folds):
        print(f"generating fold nr. {fold}")
        out_path = f"{audio_directory}/{audio_filename}_{fold}.wav"
        meta_out_path = f"{meta_directory}/{meta_filename}_{fold}.csv"
        if fold % 20 == 0:
            shuffle(packed_paths)
            print("Shuffling packed keys")

        augmentations = [
            Gain(0.5),
            AddNoise(0.5),
            # PolarityInversion(0.5)
        ]

        gen_dict = generation_dict(overlap_list, overlap_prob, sr, bpm, max_samples, midi_range)
        generation_dict_to_audio(packed_paths, sr, gen_dict, out_path, meta_out_path, max_samples, augmentations, fold, folds)


if __name__ == '__main__':
    pass
    # sr = 16000
    # # bpm = 80
    # bpm = 60
    # max_real_time_mins = 15
    #
    # # max_beats = bpm * max_real_time_mins
    #
    # dry_data_paths = "E:/Dataset/nsynth/guitar/audio/*"
    # files = glob.glob(dry_data_paths)
    # packed_paths = pack_paths(files)
    #
    # overlap_list = [1, 2]
    # overlap_prob = [20, 80]
    #
    # midi_range = [40, 42]
    #
    # max_samples = sr * 2 * 60
    #
    # gen_dict = generation_dict(overlap_list, overlap_prob, sr, bpm, max_samples, midi_range)
    # #
    # # gen_list = []
    # # start_idx = 0
    # # end_idx = 0
    # # for key in gen_dict.keys():
    # #     if gen_dict[key][0][0] == start_idx:
    # #         gen_list.append({key: gen_dict[key][0]})
    #
    # sound_factory = SoundFactory(packed_paths, sr)
    #
    # out_path = "e:/temp.wav"
    # meta_out_path = "e:/temp.csv"
    # meta_list = []
    # audio_array = np.zeros(max_samples)
    # for key_idx, key in enumerate(gen_dict.keys()):
    #     for interval_idx, interval in enumerate(gen_dict[key]):
    #         print(f"Parsing key: {key}, {key_idx / len(list(gen_dict.keys()))}, interval: {interval}, {interval_idx / len(gen_dict[key])}")
    #         y, relative_time_interval = sound_factory.get_note(key, interval)
    #
    #         interval = (int(interval[0]), int(interval[1]))
    #         audio_array[slice(*interval)] = np.add(audio_array[slice(*interval)], y)
    #         current_time = librosa.samples_to_time(interval[0], sr=sr)
    #         meta_list = append_to_meta(meta_list, key, out_path, current_time, relative_time_interval, key)
    #
    # output.write_wav(out_path, audio_array, sr)
    # df = pd.DataFrame(meta_list)
    # df.to_csv(meta_out_path, sep='\t', index=False, header=False)
    # print(f"saving to :{out_path}")
    #
    # # generation_dict_to_audio(packed_paths, sr, gen_dict, "e:/temp.wav", "e:/temp.scv")
