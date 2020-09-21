#from librosa import output, load
import librosa
from data_gen_utls import extract_sub_dir
import numpy as np

#from externals.pyagc.agc.agc import tf_agc


class SoundFactory:
    def __init__(self, packed, sr, augmentations=None):
        self.packed = packed
        self.sr = sr
        self.augmentations = augmentations

    def _detect_onset(self, y):
        try:
            frame_len = 2048
            if len(y) > frame_len:
                onset_sample = librosa.effects.split(y, 50, frame_length=frame_len)[0][0]
            else:
                onset_sample = len(y) - 1
        except:
            pass
            print("Error")
        return librosa.samples_to_time(onset_sample, self.sr)

    def _detect_offset(self, y):
        # offset_sample = librosa.effects.split(y, 50)[0][1]
        # y[offset_sample:] = 0

        try:
            frame_len = 2048
            if len(y) > frame_len:
                offset_sample = librosa.effects.split(y, 50, frame_length=frame_len)[0][1]
            else:
                offset_sample = len(y) - 1
        except:
            pass
            print("Error")
        return librosa.samples_to_time(offset_sample, self.sr)

    def get_note(self, midi, interval):
        path = self.packed[int(midi)].get()
        length = interval[1] - interval[0]
        y_length = 0
        y = np.zeros(0)
        while y_length < length:
            y_temp, sr = librosa.load(path, sr=self.sr)
            y_temp = 0.8 * librosa.util.normalize(y_temp)
            # (y_temp, D, E) = tf_agc(y_temp, sr)
            y_temp_interval = librosa.effects.split(y_temp, 50)[0]
            y_temp = y_temp[y_temp_interval[0]: y_temp_interval[1]]
            y = np.append(y, y_temp, axis=None)
            y_length += y_temp_interval[1]
        y = y[0: int(length)]

        if self.augmentations:
            for augmentation in self.augmentations:
                y = augmentation.augment(y, sr)

        start_time = self._detect_onset(y)
        end_time = self._detect_offset(y)
        return y, (start_time, end_time)

class SimpleSound:
    def __init__(self, path, midi=None, sr=16000, normalize=True):
        self.path = path
        self.y, self.sr = librosa.load(path, sr=sr)
        if normalize:
            self.y = librosa.util.normalize(self.y)
        self.onset = self._detect_onset()
        self.offset = self._detect_offset()
        if not midi:
            self.midi = extract_sub_dir(path)
        else:
            self.midi = midi

    def _detect_onset(self):
        offset_sample = librosa.effects.split(self.y, 50)[0][0]
        return librosa.samples_to_time(offset_sample, self.sr)

    def _detect_offset(self):
        offset_sample = librosa.effects.split(self.y, 50)[0][1]
        self.y[offset_sample:] = 0
        return librosa.samples_to_time(offset_sample, self.sr)


class RandomChord:
    def __init__(self, idx, midis, packed, sr=16000):
        self.idx = idx
        self.midis = midis
        self.simple_sounds = []
        self.sr = sr
        self.paths = [packed[midi].get() for midi in self.midis]
        self.name = '_'.join(str(midis)) + f"_{idx}.wav"
        self.y_out = []

    def generate_audio(self):
        self.simple_sounds = [SimpleSound(path) for path in self.paths]
        ys = [simple_sound.y for simple_sound in self.simple_sounds]
        self.y_out = [sum(y) for y in zip(*ys)]
        return self.y_out

# TODO: Legacy code: to be rewritten one day.
# class Single:
#     def __init__(self, core_path, idx):
#         self.idx = idx
#         self.core_path = core_path
#         self.core_midi = int(extract_sub_dir(core_path))
#         self.simple_sounds = []
#
#     def generate_audio(self):
#         sound_0 = SimpleSound(self.core_path)
#         self.simple_sounds.append(sound_0)
#         self.sr = sound_0.sr
#
#         self.y_out = sound_0.y
#         self.name = f"{self.core_midi}_{self.idx}.wav"
#
#
# class Double:
#
#     def __init__(self, core_path, idx, first_interval=2):
#         self.idx = idx
#         self.core_path = core_path
#         self.core_midi = int(extract_sub_dir(core_path))
#         self.first_interval_midi = self.core_midi + first_interval
#         self.first_interval_path = packed[self.first_interval_midi].get()
#         self.simple_sounds = []
#
#     def generate_audio(self):
#         sound_0 = SimpleSound(self.core_path)
#         self.simple_sounds.append(sound_0)
#         self.sr = sound_0.sr
#         sound_1 = SimpleSound(self.first_interval_path)
#         self.simple_sounds.append(sound_1)
#
#         self.y_out = sound_0.y + sound_1.y
#         self.name = f"{self.core_midi}_{self.first_interval_midi}_{self.idx}.wav"
#
#
# class Triad:
#     first_interval = 4
#     second_interval = 7
#
#     def __init__(self, core_path, idx, first_interval=2, second_interval=5):
#         self.idx = idx
#         self.core_path = core_path
#         self.core_midi = int(extract_sub_dir(core_path))
#         self.first_interval_midi = self.core_midi + first_interval
#         self.second_interval_midi = self.core_midi + second_interval
#         self.first_interval_path = packed[self.first_interval_midi].get()
#         self.second_interval_path = packed[self.second_interval_midi].get()
#         self.simple_sounds = []
#
#     def generate_audio(self):
#         sound_0 = SimpleSound(self.core_path)
#         self.simple_sounds.append(sound_0)
#         self.sr = sound_0.sr
#         sound_1 = SimpleSound(self.first_interval_path)
#         self.simple_sounds.append(sound_1)
#         sound_2 = SimpleSound(self.second_interval_path)
#         self.simple_sounds.append(sound_2)
#
#         self.y_out = sound_0.y + sound_1.y + sound_2.y
#         self.name = f"{self.core_midi}_{self.first_interval_midi}_{self.second_interval_midi}_{self.idx}.wav"


#END
