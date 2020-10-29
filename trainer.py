import pandas as pd
import torch
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import numpy as np

from audio_dataset import AudioDataset
from audio_feat_gen.configuration import TrainingConfig
from models import MultiClassifier, OneChannelResnet
from audio_feat_gen.utils import run_confusion_matrix

# from torchsummary import summary


def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)


def append_stats(stats, epoch_stats):
    stats["loss"].append(epoch_stats["loss"])
    stats["acc"].append(epoch_stats["acc"])
    stats["f1"].append(epoch_stats["f1"])
    return stats


def save_stats(epoch_idx, stats, config, model, optimizer,chpt_folder, experiment_name):
    to_save = {
        "config": config,
        "epoch": epoch_idx,
        "stats": stats,
        "state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()  # ,
    }
    torch.save(to_save, f"{chpt_folder}/{experiment_name}/checkpoint_epoch{epoch_idx}.pth")


def train(starting_epoch_idx, epochs, model, criterion, optimizer, train_dataloader, eval_dataloader, device, stats, config, chpt_folder, experiment_name):
    for epoch_idx in range(starting_epoch_idx, starting_epoch_idx + epochs):
        print(f"Epoch: {epoch_idx} - STARTING")
        epoch_stats = fit_model(epoch_idx, model, train_dataloader, optimizer, criterion, device=device)
        print(f"Training: Epoch {epoch_idx} average: loss: {epoch_stats['loss']}, acc: {epoch_stats['acc']}, f1: {epoch_stats['f1']}")

        append_stats(stats, epoch_stats)
        # val_l, val_a = fit_model(i, model, valid_dataloader, phase = 'validation')

        eval_epoch_stats = fit_model(1, model, eval_dataloader, optimizer, criterion, phase='validataion', device=device)
        print(f"Evaluation: Epoch {epoch_idx} average: loss: {eval_epoch_stats['loss']}, acc: {eval_epoch_stats['acc']}, f1: {eval_epoch_stats['f1']}")
        stats["eval_stats"][epoch_idx] = eval_epoch_stats

        save_stats(epoch_idx, stats, config, model, optimizer, chpt_folder, experiment_name)


def fit_model(epoch, model, dataloader, optimizer, criterion, phase='training', device="cpu"):
    if phase == 'training':
        model.train()
    elif phase == 'validataion':
        model.eval()

    running_loss = []
    running_acc = []
    running_prec = []
    running_rec = []
    running_f1 = []
    confusion_mtrx = np.zeros((len(dataloader.dataset.labels_dict.keys()), 2, 2))

    dataloader.dataset.load_fold(epoch)

    for i, (inputs, target) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1)  # add channels simention when mono

        inputs, target = inputs.to(device), target.to(device)

        if phase == 'training':
            optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs.double(), target.double())
        running_loss.append(loss.item())

        if phase == 'training':
            loss.backward()
            optimizer.step()

        f1 = f1_score(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()), average="weighted")
        running_f1.append(np.asarray(f1).mean())

        prec = precision_score(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()), average="weighted")
        running_prec.append(np.asarray(prec).mean())

        rec = recall_score(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()), average="weighted")
        running_rec.append(np.asarray(rec).mean())

        acc = pred_acc(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()))
        running_acc.append(np.asarray(acc).mean())

        if phase == 'validataion':
            new_conf_mtr = multilabel_confusion_matrix(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()))
            confusion_mtrx = np.add(confusion_mtrx, new_conf_mtr)

        total_batch_loss = np.asarray(running_loss).mean()
        total_batch_acc = np.asarray(running_acc).mean()
        total_batch_f1 = np.asarray(running_f1).mean()

        print(f"{phase} | Epoch: {epoch}, batch: {i}/{len(dataloader)}:  loss: {total_batch_loss}, {phase} acc: {total_batch_acc}, {phase} f1: {total_batch_f1}")

    return {
        'loss': total_batch_loss,
        "acc": total_batch_acc,
        "f1": total_batch_f1,
        "prec": np.asarray(running_prec).mean(),
        "rec": np.asarray(running_rec).mean(),
        "conf_mtrx": confusion_mtrx
    }


def get_dataloaders(feat_folder, class_labels, folds, batch_size):

    data_set = AudioDataset(feat_folder, class_labels, folds)

    train_dl = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return train_dl
