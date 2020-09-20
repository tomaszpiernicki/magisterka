import torch
import matplotlib.pyplot as plt

import numpy as np

def plot_checkpoint(checkpoint, n_max = 5):
    if type(checkpoint) is str:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    hop = 4980

    losses = checkpoint["stats"]["loss"][hop:]
    acc = checkpoint["stats"]["acc"][hop:]
    f1 = checkpoint["stats"]["f1"][hop:]

    fig, axs = plt.subplots(3)
    axs[0].plot(losses)
    axs[1].plot(acc)
    axs[2].plot(f1)

    axs[0].set_title("loss")
    axs[1].set_title("acc")
    axs[2].set_title("f1")

    fig.subplots_adjust(hspace=1.2)
    print(checkpoint["epoch"])
    print(f"Entries: {len(losses)}")
    max_f1 = np.argpartition(f1, -n_max)[-n_max:]
    [print(f"idx: {idx}, f1: {f1[idx]}, acc: {acc[idx]}") for idx in max_f1]
    plt.show()

if __name__ == "__main__":
    plot_checkpoint("i:\magisterka\checkpoints\\resnet_quad_3_notes_v4\checkpoint_epoch6366.pth")