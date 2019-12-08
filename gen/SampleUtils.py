import numpy as np

from gen import Note


class SampleUtils:
    bpm = 100
    period = 60.0 / bpm

    def full_service(self, note: Note):
        self.clean_up(note)
        self.fit_sample_size(note)
        self.fade_out(note)

    def fit_sample_size(self, note: Note):
        multiplier = note.sound_length.value
        real_length = multiplier * self.period
        y_real_length = real_length * note.sound_br

        if len(note.sound) >= y_real_length:
            note.sound = note.sound[:int(y_real_length)]
        else:
            np.pad(note.sound, (0, y_real_length - len(note.sound)))

    def fade_out(self, note: Note, duration=4):
        n = len(note.sound) // duration
        y_to_operate = note.sound[-n:]

        b = 1
        a = -b / n

        for i in range(0 , n):
            y_to_operate[i] *= (a * i + b)

        note.sound = note.sound[: -n]
        note.sound = np.concatenate([note.sound, y_to_operate])

    def clean_up(self, note, silence_level=0.02):
        y_max_inx = np.argmax(note.sound)
        temp_y = note.sound[:y_max_inx]
        temp_y = temp_y[::-1]
        y_start = None
        for i, a in enumerate(temp_y):
            if abs(temp_y[i]) <= silence_level:
                y_start = len(temp_y) - i
                break
        note.sound = note.sound[y_start:]