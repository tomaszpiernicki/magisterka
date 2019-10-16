import os
import re
from re import match
from random import randrange

from gen.Note import Note
from gen.NoteId import NoteId


class NoteFactory:

    path_to_samples = "/home/piernik/Studia/bazy_wiedzy/guitar"

    def generateRandom(self, quarters_to_fill, notes_stereotypes):

        file_paths = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.path_to_samples):
            for file in f:
                if '.mp3' in file:
                    file_paths.append(os.path.join(r, file))

        i = 0
        notes = []
        while i < quarters_to_fill:
            i += 1
            randNumer = randrange(len(file_paths))
            rand_audio = file_paths[randNumer]
            note_id = self.extract_first_noteId(rand_audio)
            note_id = note_id[1:-1]
            note = Note(note_id, rand_audio, notes_stereotypes[0])
            notes.append(note)
        return notes

    def extract_first_noteId(self, rand_audio):
        try:
            result = re.findall('_[CDEFGAB]s?[0123456789]?_' , rand_audio)[0]
        except Exception as IndexError:
            result = ''
        return result