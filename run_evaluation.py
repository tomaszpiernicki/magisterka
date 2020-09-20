import os
import re

import torch
from torch import nn, optim

from configuration import EvalConfig
from models import MultiClassifier, OneChannelResnet
from utils import run_confusion_matrix, plot_adjacent_bars, plot_stacked_bars

from trainer import get_dataloaders, fit_model

import numpy as np

def run_evaluation(config_file, from_file=None):
    configuration = EvalConfig()
    configuration.parse_config(config_file)

    class_labels = configuration.class_labels
    # device = configuration.device
    device = "cpu"
    experiment_name = configuration.experiment_name
    restart_checkpoint = configuration.restart_checkpoint
    batch_size = configuration.batch_size
    feat_folder = configuration.feature_folder
    folds = configuration.folds
    eval_dataloader = get_dataloaders(feat_folder, class_labels, folds, batch_size)
    outputs = configuration.outputs
    checkpoints = configuration.checkpoints

    if from_file:
        checkpoints_evaluaton = torch.load(from_file)
    else:
        checkpoints_evaluaton = {}

        for idx, checkpoint in enumerate(checkpoints):
            epoch = int(re.findall('[0123456789]+', checkpoint)[-1])

            print(f"Evaluation: {idx + 1}/{len(checkpoints)}")
            print(f"Using device: {device}")
            model = OneChannelResnet(class_labels, pretrained=False)
            model.to(device)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters())

            print(f"Loading from checkpoint: {checkpoint}")
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            stats = fit_model(1, model, eval_dataloader, optimizer, criterion, phase='validataion', device=device)
            print(f"average: acc: {stats['acc']}, f1: {stats['f1']}, prec: {stats['prec']}, recall: {stats['rec']}")
            checkpoints_evaluaton[epoch] = stats

        torch.save(checkpoints_evaluaton, f"{outputs}{experiment_name}.pth")
        print(f"saved: {outputs}{experiment_name}")


    epoch_dict = {}
    for epoch in checkpoints_evaluaton.keys():
        epoch_dict[epoch] = checkpoints_evaluaton[epoch]['f1']
    epoch_dict = {k: v for k, v in sorted(epoch_dict.items(), key=lambda item: item[1])}.keys()
    best_epoch = list(epoch_dict)[-1]

    best_checkpoint = checkpoints_evaluaton[best_epoch]
    print(f"Best is: {best_epoch} | average: acc: {best_checkpoint['acc']}, f1: {best_checkpoint['f1']}, prec: {best_checkpoint['prec']}, recall: {best_checkpoint['rec']}")
    conf_mtrx = checkpoints_evaluaton[best_epoch]['conf_mtrx']
    conf_bar = np.reshape(conf_mtrx, (conf_mtrx.shape[0], 4))

    plot_stacked_bars(conf_bar, list(class_labels.keys()))
    # plot_adjacent_bars(conf_bar)
    # run_confusion_matrix(conf_mtrx, eval_dataloader.dataset.labels_dict.keys())


if __name__ == "__main__":
    run_evaluation("configs/validating/valid_resnet_quad_notes_v0.4.json", from_file="e:\\Dataset\\chords\\0.4\\quad_many_valid\\evaluations\\valid_resnet_quad_notes_v0.4.pth")
    # run_evaluation("configs/validating/valid_resnet_triple_notes_v0.4.json") #, from_file="e:\\Dataset\\chords\\0.4\\triple_many_valid\\evaluations\\valid_resnet_triple_notes_v0.4.pth")
    # run_evaluation("configs/validating/valid_resnet_double_notes_v0.4.json", from_file="e:\\Dataset\\chords\\0.4\\double_many_valid\\evaluations\\valid_resnet_double_notes_v0.4.pth")
    # run_evaluation("configs/validating/valid_resnet_single_notes_v0.4.json", from_file='e:\\Dataset\\chords\\0.4\\single_many_valid\\evaluations\\valid_resnet_single_notes_v0.4.pth')

# END
