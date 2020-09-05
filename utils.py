import os
import random
import re

import numpy as np

import config
import glob

import matplotlib.pyplot as plt

eps = np.finfo(np.float).eps


def create_folder(_fold_path):
    if not os.path.exists(_fold_path):
        os.makedirs(_fold_path)


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def split_multi_channels(data, num_channels):
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = int(in_shape[2] / num_channels)
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()
    return tmp


def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((int(data.shape[0] / subdivs), subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((int(data.shape[0] / subdivs), subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] / subdivs, subdivs, data.shape[1], data.shape[2]))
    return data


def extract_sub_dir(path):
    try:
        result = re.findall('-[0123456789][0123456789][0123456789]-', path)[0]
    except Exception as IndexError:
        result = ''
    return result[1:-1]


def pack_paths(files):
    path_dict = {}

    for file in files:
        k = int(extract_sub_dir(file))
        if k not in path_dict:
            path_dict[k] = RoundRobin()
        path_dict[k].add(file)

    return path_dict


class RoundRobin():
    def __init__(self):
        self.lst = []

    def get(self):
        to_return = self.lst[0]
        self.lst = self.lst[1:]
        self.add(to_return)
        return to_return

    def add(self, path):
        self.lst.append(path)

    def reverse(self):
        self.lst.reverse()

    def shuffle(self):
        random.shuffle(self.lst)

    def remove(self, index):
        print("Not implemented yet.")

    def __len__(self):
        return len(self.lst)

def pop_many(lst, n):
    newlist = lst[:n]
    del lst[:n]
    return newlist


def match_lists_by_len(list_of_lists):
    lengths = [len(lst) for lst in list_of_lists]
    max_len = max(lengths)
    for lst in list_of_lists:
      len_diff = max_len - len(lst)
      lst = np.pad(lst, (0, len_diff), "constant")
      # lst.extend([0]*len_diff)
    return list_of_lists


def plot_vaves():
    files = glob.glob(config.dry_data_path)
    print(len(files))
    packed = pack_paths(files)

    tmp = list(packed.keys())
    print(config.class_labels)
    print(tmp)
    for key in tmp:
        if str(key) not in config.class_labels.keys():
            packed.pop(key)
    files_histogram = [len(rr) for rr in packed.values()]
    print(len(packed.keys()))
    plt.bar(packed.keys(), files_histogram)
    plt.show()

def plot_featues():
    feat_file = "E:/Dataset/chords/0.3/quintiple/features/mbe_mon_fold11_467.npz"
    dmp = np.load(feat_file, allow_pickle=True)
    data, label = dmp['arr_0'], dmp['arr_1']
    data = np.transpose(data)
    label = np.expand_dims(label, (1))

    for i, x in enumerate(label):
        if x != 0:
            print(i)

    fig, (ax1, ax2) = plt.subplots(1, 2) #, gridspec_kw={'width_ratios': [20, 1]})
    ax1.imshow(data)
    ax2.imshow(label)

    # ax1.set_figheight(5)
    # ax1.set_figwidth(15)
    plt.show()
    pass


if __name__ == "__main__":
    # Histogram of aviable sample files
    plot_featues()