import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from gen.SilenceTools import SilenceTools
from gen.Note import Note
from gen.NoteFactory import NoteFactory
from gen.NoteLength import NoteLength

def generate_note(note: Note, period):
    y, sr = librosa.load(note.sound_file)
    y = SilenceTools.cleanUp(y)
    multiplier = note.sound_length.value[0]
    real_length = multiplier * period
    y_real_length = real_length * sr

    if(len(y) >= y_real_length ):
        y = y[:int(y_real_length)]
    else:
        np.pad(y, (0, y_real_length - len(y)))

    y = fade_out(y)

    return y, sr

def fade_out( y, duration = 4 ):
    n = len(y) // duration
    y_to_operate = y[ -n :]

    b = 1
    a = -b / n

    for i in range(0 , n):
        y_to_operate[i] *= (a * i + b)

    y = y[: -n]
    y = np.concatenate([y, y_to_operate])
    return y

def generate_notes_array():
    bpm = 100
    period = 60 / bpm
    note = NoteLength.Quarter
    notes_stereotypes = [note]
    metrum = [4, 4]
    tacts_to_generate = 2

    quarter_real_time = 60 / bpm
    quarters_in_tact = 4
    quarters_to_fill = quarters_in_tact * tacts_to_generate

    noteFactory = NoteFactory()
    notes_array = noteFactory.generate_random(quarters_to_fill, notes_stereotypes)

    y = []
    sr = []

    for note in notes_array:
        y_temp, sr_temp = generate_note(note, period)
        y.append(y_temp)
        sr.append(sr_temp)

    merged_list = []
    for arr in y:
        merged_list += list(arr)

    merged_list = np.asarray(merged_list)

    plt.figure()

    S = librosa.feature.melspectrogram(merged_list, sr[0])
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    from os.path import expanduser
    path_to_output = expanduser("~") + "/generator/"

    plt.savefig(path_to_output + "plot.png")

    librosa.output.write_wav(path_to_output + "output.wav", merged_list, sr=sr[0])

    return
