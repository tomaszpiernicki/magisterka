from gen import NoteLength


class Note:

    def __init__(self, sound_id, sound_file, sound_length: NoteLength):
        self.sound_id = sound_id
        self.sound_length = sound_length
        self.sound_file = sound_file
