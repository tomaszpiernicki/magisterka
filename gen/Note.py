from numba.cuda import selp

from gen import NoteLength, NoteFactory

class Note:

    sound = []
    sound_br = 0

    def __init__(self, sound_id, sound_file, sound_length: NoteLength):
        self.sound_id = sound_id
        self.sound_length = sound_length
        self.sound_path = sound_file

    def set_sound(self, sound, br):
        self.sound = sound
        self.sound_br = br
