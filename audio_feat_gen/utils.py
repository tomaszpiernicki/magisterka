import os
import numpy as np

from data_gen_utls import pack_paths
#from old import config
import glob

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

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


def pop_many(lst, n):
    newlist = lst[:n]
    del lst[:n]
    return newlist


def match_lists_by_len(list_of_lists):
    lengths = [len(lst) for lst in list_of_lists]
    max_len = max(lengths)
    for idx, lst in enumerate(list_of_lists):
        len_diff = max_len - len(lst)
        list_of_lists[idx] = np.pad(lst, (0, len_diff), "constant")
    return list_of_lists


#def plot_vaves():
#    files = glob.glob(config.dry_data_path)
#    print(len(files))
#    packed = pack_paths(files)
#
    #tmp = list(packed.keys())
    #print(config.class_labels)
    #print(tmp)
    #for key in tmp:
    #    if str(key) not in config.class_labels.keys():
    #        packed.pop(key)
    #files_histogram = [len(rr) for rr in packed.values()]
    #print(len(packed.keys()))
    #plt.bar(packed.keys(), files_histogram)
    #plt.show()


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


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=6):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d",  ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    # axes.set_xlabel('True label')
    # axes.set_ylabel('Predicted label')
    axes.set_title(class_label, fontsize=fontsize)


def run_confusion_matrix(vis_arr, labels):

    rows = 8
    fig, ax = plt.subplots(rows, round(len(labels)/rows), figsize=(19, 9))

    for idx, (axes, cfs_matrix, label) in enumerate(zip(ax.flatten(), vis_arr, labels)):
        print(f"Plotting: {idx+1}/{len(labels)}")
        print_confusion_matrix(cfs_matrix.astype(np.int32), axes, label, ["T", "F"])

    fig.tight_layout()
    plt.show()


def plot_adjacent_bars(data):

    data = [xs[0] for xs in data]
    x = list(range(0, len(data)))


    # dx = (np.arange(data.shape[1])-data.shape[1]/2.)/(data.shape[1]+2.)
    # d = 1./(data.shape[1]+2.)


    fig, ax=plt.subplots()
    # for i in range(data.shape[1]):
    #     ax.bar(x+dx[i],data[:,i], width=d, label="label {}".format(i))

    ax.bar(x, data)
    plt.legend(framealpha=1)
    plt.show()


def plot_stacked_bars(data, labels: list = None):
    x = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    fig.figsize=(20, 3)
    for i in range(data.shape[1]):
        bottom = np.sum(data[:, 0:i], axis=1)
        ax.bar(x, data[:, i], bottom=bottom, label="label {}".format(i), align='edge', width=0.8)

    height = np.sum(data, 1)[0]
    ax.set_ylim(bottom=height-115, top=height+5)
    ax.set_xlabel("klucze midi")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, fontsize=8)
    plt.legend(["TN", "FP", "FN", "TP"], framealpha=1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Histogram of aviable sample files
    plot_featues()

    # run_confusion_matrix(conf_mtr_path)
