import os
import string
import threading
from zipfile import ZipFile
from PIL import Image, ImageColor, ImageChops
import pylab
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np



from gen.NoteLength import NoteLength

pylab.matplotlib.use('Agg')  # No pictures displayed

from sphinx.util import requests
from os.path import expanduser

class SampleFileManager:
    file_paths = []
    path_to_output = expanduser("~") + "/generator/"
    path_to_samples = path_to_output + "/samples_trimmed/"
    path_to_charts = path_to_output + "/charts/"
    path_to_audio = path_to_output + "/audio/"

    def k__init__(self):
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

    def save_spectogram(self, y, sr, out_name, idx):
        path_to_fig = self.path_to_charts + out_name[:-1]
        hop_length = 512
        n_fft = 1024

        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        D_DB = librosa.amplitude_to_db(D, ref=np.max)

        length = librosa.core.get_duration(y, sr)
        length = length / 4.8
        real_lenth = length * 6.4
        plt.figure(frameon=False, figsize=(real_lenth, 4.8))
        plt.autoscale(False)
        librosa.display.specshow(D_DB, sr=sr, hop_length=hop_length)

        if not os.path.exists(path_to_fig):
            os.makedirs(path_to_fig)

        image_name = path_to_fig + "/" + str(idx)
        plt.savefig(image_name + ".jpg")
        plt.close('all')

        self.split_img_vertically(image_name, 80)
        # librosa.output.write_wav( self.path_to_audio + out_name + ".wav", y , sr)

    def split_img_vertically(self, image_path,  window_width = 100):
        im = Image.open(image_path + ".jpg")
        im = self.trim_border(im)
        im = self.white2black(im)
        width, height = im.size
        temp_im_path = os.path.splitext(image_path)[0]
        padding = window_width - width % window_width
        im1 = self.add_margin(im, padding)
        # im1.save(image_path + "padding.jpg", "jpeg")
        for i in range(0, width, window_width):
            temp_im = im1.crop((i, 0, i + window_width, height))
            temp_im_path = temp_im_path + "_"
            temp_im.save(temp_im_path + ".jpg", "jpeg")
        os.remove(image_path + ".jpg")

    def add_margin(self, pil_img, right, top=0, bottom=0, left=0, color=0):
        width, height = pil_img.size
        new_width = width + right + left - 2
        new_height = height + top + bottom - 2
        result = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        result.paste(pil_img, (left, top), mask=pil_img.split()[3])
        return result

    def trim_border(self, im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    def white2black(self, im, ):
        im = im.convert('RGBA')

        data = np.array(im)  # "data" is a height x width x 4 numpy array
        red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

        # Replace white with red... (leaves alpha values alone...)
        white_areas = (red == 255) & (blue == 255) & (green == 255)
        data[..., :-1][white_areas.T] = (0, 0, 0)  # Transpose back needed

        im2 = Image.fromarray(data)
        return im2
