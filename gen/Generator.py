from gen.AudioFileManager import SampleFileManager
from gen.NoteFactory import NoteFactory
from gen.NoteLength import NoteLength
import numpy as np

def generate_random_melody():
    bpm = 100
    period = 60 / bpm
    notes_stereotypes = [NoteLength.Quarter,  NoteLength.Quarter,  NoteLength.Quarter,  NoteLength.Quarter,  NoteLength.Quarter,  NoteLength.Quarter, NoteLength.Quarter , NoteLength.Quarter]
    metrum = [4, 4]
    tacts_to_generate = 2
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

    sfm = SampleFileManager()
    sfm.save_spectogram(np.asarray(audio), notes[0].sound_br, audio_name)

    # librosa.output.write_wav(path_to_output + "output.wav", merged_list, sr=sr[0])
