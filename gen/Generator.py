import random

from gen.AudioFileManager import SampleFileManager
from gen.NoteFactory import NoteFactory
from gen.NoteLength import NoteLength
import numpy as np

def generate_random_melody(idx, bpm=100, metrum=None):
    if metrum is None:
        metrum = [4, 4]
    period = 60 / bpm

    all = [NoteLength.Whole, NoteLength.Half, NoteLength.Quarter, NoteLength.Eighth, NoteLength.Sixteenth]
    notes_stereotypes = [random.choice(all)]
    # notes_stereotypes =  [NoteLength.Quarter,  NoteLength.Quarter,  NoteLength.Quarter,  NoteLength.Quarter, NoteLength.Quarter,  NoteLength.Quarter, NoteLength.Quarter , NoteLength.Quarter]
    quarters_in_tact = 4
    notes = []

    nf = NoteFactory()
    for stereotype in notes_stereotypes:
        notes.append(nf.get_note(stereotype))

    audio = []
    audio_name = ""
    for note in notes:
        for sound in note.sound:
            audio.append(sound)
        audio_name += note.sound_id + "_"

    audio_name = audio_name[:-1]

    multiplier = NoteLength.Sixteenth.value
    real_length = multiplier * period
    y_real_length = real_length * notes[0].sound_br
    frames_number = ceildiv(len(audio), y_real_length)

    sfm = SampleFileManager()
    sfm.save_spectogram(np.asarray(audio), notes[0].sound_br, audio_name, idx)

    # librosa.output.write_wav(path_to_output + "output.wav", merged_list, sr=sr[0])

def ceildiv(a, b):
    return -(-a // b)