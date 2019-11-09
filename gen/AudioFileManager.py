import os
import string
import threading
from zipfile import ZipFile
import pylab
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
pylab.matplotlib.use('Agg')  # No pictures displayed

from sphinx.util import requests
from os.path import expanduser

class SampleFileManager:
    file_paths = []
    path_to_output = expanduser("~") + "/generator/"
    path_to_samples = path_to_output + "/samples_trimmed/"
    path_to_charts = path_to_output + "/charts/"
    path_to_audio = path_to_output + "/audio/"

    def __init__(self):
        self.lock = threading.Lock()

    def get_sample_from_file(self, sound_file: string):
        #with self.lock:
        y, sr = librosa.load(sound_file)
        return y, sr

    def get_aviable_samples(self):
        self.lock.acquire()
        try:
            if not self.file_paths:
                self.file_paths = self.get_file_paths()
        finally:
            self.lock.release()
        return self.file_paths

    def save_audio(self):
        with self.lock:
            # TODO
            return

    def get_file_paths(self, download=False):

        if os.listdir(self.path_to_samples):
            for r, d, f in os.walk(self.path_to_samples):
                for file in f:
                    if '.mp3' in file:
                        self.file_paths.append(os.path.join(r, file))
            return self.file_paths

        url = 'https://www.philharmonia.co.uk/assets/audio/samples/guitar/guitar.zip'
        zipname = "plik"
        r = requests.get(url)

        with open(zipname, 'wb') as f:
            f.write(r.content)

        with ZipFile(zipname, 'r') as zipObj:
            zipObj.extractall(self.path_to_samples)

        os.remove(zipname)

        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.path_to_samples):
            for file in f:
                if '.mp3' in file:
                    self.file_paths.append(os.path.join(r, file))

        return self.file_paths

    def save_spectogram(self, y, sr, out_name):
        path_to_fig = self.path_to_charts + out_name

        hop_length = 512
        n_fft = 1024
        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        D_DB = librosa.amplitude_to_db(D, ref=np.max)
        #plt.axis("off")
        librosa.display.specshow(D_DB, sr=sr, hop_length=hop_length)
        plt.savefig(path_to_fig)
        librosa.output.write_wav( self.path_to_audio + out_name + ".wav", y , sr)