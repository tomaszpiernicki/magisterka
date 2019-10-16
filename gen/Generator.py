import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from gen import SilenceTools
from gen.Note import Note
from gen.NoteFactory import NoteFactory
from gen.NoteLength import NoteLength


def generate_note(note: Note):
    y, sr = librosa.load(note.sound_file)
    y = SilenceTools.SilenceTools.cleanUp(y)
    multiplier = note.sound_length.value[0]
    real_length = multiplier * period
    y_real_length = real_length * sr

    if(len(y) >= y_real_length ):
        y = y[:int(y_real_length)]
    else:
        np.pad(y, (0, y_real_length - len(y)))

    print(real_length)
    return y, sr

bpm = 100
period = 60 / bpm
note = NoteLength.Quarter
notes_stereotypes = [note]
metrum = [4, 4]
tacts_to_generate = 2

quarter_real_time = 60 / bpm
print("querter_real_time: ", quarter_real_time)
quarters_in_tact = 4
quarters_to_fill = quarters_in_tact * tacts_to_generate

noteFactory = NoteFactory()
notes_array = noteFactory.generateRandom(quarters_to_fill, notes_stereotypes)

y = []
sr = []

for note in notes_array:
    y_temp, sr_temp = generate_note(note)
    y.append(y_temp)
    sr.append(sr_temp)

merged_list = []
for arr in y:
    merged_list += list(arr)

merged_list = np.asarray(merged_list)

plt.figure()
librosa.display.waveplot(merged_list, sr=sr[0])
plt.savefig("plot.png")

librosa.output.write_wav("output.wav", merged_list, sr=sr[0])