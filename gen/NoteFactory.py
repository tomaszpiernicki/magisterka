import re
from random import randrange
from gen.AudioFileManager import SampleFileManager
from gen.Note import Note
from gen.SampleUtils import SampleUtils


class NoteFactory:

    def __init__(self):
        self.sfm = SampleFileManager()
        self.su = SampleUtils()

    def get_note(self, note_length) -> Note:
        rand_audio_path = self.get_random_audio_path()
        note_id = self.extract_note_id(rand_audio_path)
        note = Note(note_id, rand_audio_path, note_length)
        y, sr = self.sfm.get_sample_from_file(note.sound_path)
        note.set_sound(y, sr)
        self.su.full_service(note)
        return note

    def extract_note_id(self, rand_audio):
        try:
            result = re.findall('_[CDEFGAB]s?[0123456789]?_' , rand_audio)[0]
        except Exception as IndexError:
            result = ''
        return result[1:-1]

    def get_random_audio_path(self):
        file_paths = self.sfm.get_aviable_samples()
        rand_number = randrange(len(file_paths))
        return file_paths[rand_number]