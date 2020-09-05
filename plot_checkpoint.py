import torch
import matplotlib.pyplot as plt

import numpy as np

def plot_checkpoint(checkpoint, n_max = 5):
    if type(checkpoint) is str:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    losses = checkpoint["stats"]["loss"]
    acc = checkpoint["stats"]["acc"]
    f1 = checkpoint["stats"]["f1"]
    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(losses)
    axs[1].plot(acc)
    axs[2].plot(f1)

    print(checkpoint["epoch"])
    print(f"Entries: {len(losses)}")
    max_f1 = np.argpartition(f1, -n_max)[-n_max:]
    [print(f"idx: {idx}, f1: {f1[idx]}") for idx in max_f1]
    plt.show()

if __name__ == "__main__":
    plot_checkpoint("E:/checkpoints/resnet_double_3_notes_v1/checkpoint_epoch1084.pth")