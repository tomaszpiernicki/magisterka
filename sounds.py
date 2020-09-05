from librosa import output, load
import librosa

from utils import extract_sub_dir

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

class SimpleSound:
    def __init__(self, path, midi=None, sr=16000):
        self.path = path
        self.y, self.sr = load(path, sr=sr)
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

#END