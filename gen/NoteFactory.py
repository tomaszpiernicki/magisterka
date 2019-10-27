import os
import re
import urllib
from random import randrange
from zipfile import ZipFile

import requests

from gen.Note import Note


class NoteFactory:

    from os.path import expanduser
    path_to_samples = expanduser("~") + "/generator/samples"

    def generate_random(self, beats_to_fill, notes_stereotypes):

        file_paths = self.get_file_paths()

        i = 0
        notes = []
        while i < beats_to_fill:
            i += 1
            rand_number = randrange(len(file_paths))
            rand_audio = file_paths[rand_number]
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

    def get_file_paths(self):

        url = 'https://www.philharmonia.co.uk/assets/audio/samples/guitar/guitar.zip'
        zipname = "plik"

        r = requests.get(url)

        with open(zipname, 'wb') as f:
            f.write(r.content)

        with ZipFile(zipname, 'r') as zipObj:
            zipObj.extractall(self.path_to_samples)

        os.remove(zipname)

        file_paths = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.path_to_samples):
            for file in f:
                if '.mp3' in file:
                    file_paths.append(os.path.join(r, file))

        return file_paths
