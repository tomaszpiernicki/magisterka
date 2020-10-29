import sys

import torch
from torch import nn, optim

sys.path.insert(0, "./audio_feat_gen/")

from audio_feat_gen.configuration import TrainingConfig, EvalConfig
from models import OneChannelResnet
from trainer import get_dataloaders, train


def get_model(model_name, device, class_labels):
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    if model_name == "OneChannelResnet":
        model = OneChannelResnet(class_labels, pretrained=False)
    else:
        raise NotImplementedError()
    model.to(device)
    return model

def get_criterion(criterion_name):
    if criterion_name == "BCELoss":
        criterion = nn.BCELoss()
    elif criterion_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()  # Use this when there is no sigmoid before models output
    else:
        raise NotImplementedError()
    print(f"Loss fnc: {criterion}")
    return criterion

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters())
    print(f"Loss fnc: {optimizer}")
    return optimizer


def load_checkpoint(checkpoint, model, device, optimizer):
    print(f"Loading from checkpoint: {checkpoint}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    # TODO: add key check for whole checkpoint and stats
    if not stats.get("eval_stats", None):
        stats["eval_stats"] = {}

    starting_epoch_idx = checkpoint["epoch"] + 1

    return stats, starting_epoch_idx


def run_training(config_file, eval_config_file):
    configuration = TrainingConfig()
    configuration.parse_config(config_file)
    eval_configuration = EvalConfig()
    eval_configuration.parse_config(eval_config_file)

    class_labels = configuration.class_labels
    device = configuration.device
    restart_checkpoint = configuration.restart_checkpoint
    chpt_folder = configuration.chpt_folder
    epochs = configuration.epochs
    batch_size = configuration.batch_size
    feat_folder = configuration.feature_folder
    folds = configuration.folds
    experiment_name = configuration.experiment_name

    train_dataloader = get_dataloaders(feat_folder, class_labels, folds, batch_size)
    eval_dataloader = get_dataloaders(eval_configuration.feature_folder, eval_configuration.class_labels, eval_configuration.folds, eval_configuration.batch_size)
    
    model = get_model("OneChannelResnet", device, class_labels)
    criterion = get_criterion("BCELoss")
    optimizer = get_optimizer(model)

    stats = {"loss": [], "f1": [], "acc": [], "eval_stats" : {}}

    if restart_checkpoint:
        stats, starting_epoch_idx = load_checkpoint(restart_checkpoint, model, device, optimizer)

    if False:
        starting_epoch_idx = 0

    train(starting_epoch_idx, epochs, model, criterion, optimizer, train_dataloader, eval_dataloader, device, stats,
          configuration.config, chpt_folder, experiment_name)

if __name__ == "__main__":



    # run_training("E:\\Dataset\\transcription-crnn\\configs\\training\\resnet_test_notes.json", eval_config_file="E:\\Dataset\\transcription-crnn\\configs\\validating\\valid_resnet_test_notes.json")
    run_training("E:\\Dataset\\magisterka\\configs\\training\\resnet_double_notes_v0.5.json", eval_config_file='E:\\Dataset\\magisterka\\configs\\validating\\valid_resnet_double_notes_v0.5.json')

# END