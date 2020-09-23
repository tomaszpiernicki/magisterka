# TODO: Add augmentations support:
# #  Tłumienie + - 6
# #  Wzmacnianie
# #  Przesuwanie
# #  dodawanie szumów:
# #       White noise
import librosa
import random
import numpy as np
import random

class Augmentation:
    def __init__(self, propability):
        self.probability = propability

    def enabled(self):
        return random.random() < self.probability

    def augment(self, x, sr):
        if self.enabled():
            x = self.augment_fnc(x, sr)
        return x

    def augment_fnc(self, x, sr):
        raise NotImplementedError()


class Gain(Augmentation):
    def augment_fnc(self, y, sr):
        gain_range = (-6, 6)
        gain = random.sample(range(*gain_range), 1)[0]
        amp_mul = librosa.db_to_amplitude(gain)
        return y * amp_mul


class AddNoise(Augmentation):
    def augment_fnc(self, y, sr):
        noise_gain = random.uniform(0.05, 0.001)
        noise = noise_gain * np.random.normal(0, 1, len(y))
        return y + noise


class PolarityInversion(Augmentation):
    def augment_fnc(self, x, sr):
        return -1 * x

if __name__ == "__main__":
    g = Gain()
    g.augment_fnc(None, 16000)
