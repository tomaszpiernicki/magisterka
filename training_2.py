import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

import numpy as np

from audio_dataset import AudioDataset
from configuration import TrainingConfig
from models import MultiClassifier, OneChannelResnet

def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)

def pred_prec(original, predicted):
    pass

def fit_model(epochs, model, dataloader, optimizer, criterion, phase='training', volatile=False, device="cpu"):
    if phase == 'training':
        model.train()
    elif phase == 'validataion':
        model.eval()
        volatile = True

    running_loss = []
    running_acc = []
    running_f1 = []

    dataloader.dataset.load_fold(epochs)

    for i, (inputs, target) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1)  # add channels simention when mono

        inputs, target = inputs.to(device), target.to(device)

        if phase == 'training':
            optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs.double(), target.double())
        running_loss.append(loss.item())

        f1 = f1_score(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()), average="weighted")
        running_f1.append(np.asarray(f1).mean())

        acc = pred_acc(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()))
        running_acc.append(np.asarray(acc).mean())

        # conf_matrix = multilabel_confusion_matrix(target.to('cpu').detach(), torch.round(outputs.to('cpu').detach()), dataloader.dataset.labels_dict.keys())

        if phase == 'training':
            loss.backward()
            optimizer.step()

        total_batch_loss = np.asarray(running_loss).mean()
        total_batch_acc = np.asarray(running_acc).mean()
        total_batch_f1 = np.asarray(running_f1).mean()

        print(f"{phase} | Epoch: {epochs}, batch: {i}/{len(dataloader)}:  loss: {total_batch_loss}, {phase} acc: {total_batch_acc}, {phase} f1: {total_batch_f1}")

    return total_batch_loss, total_batch_acc, total_batch_f1


def get_dataloaders(feat_folder, class_labels, folds, batch_size):

    # TODO think about it!
    data_set = AudioDataset(feat_folder, class_labels, folds)
    # data_set_len = len(data_set)
    # train_test_ratio = 0.1
    #
    # train_set, eval_set = torch.utils.data.random_split(data_set, [int(data_set_len * train_test_ratio),
    #                                                               int(data_set_len * (1 - train_test_ratio))])

    train_dl = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # eval_dl = DataLoader(
    #     eval_set,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0
    # )

    return train_dl #, eval_dl


def run_training(config_file):
    configuration = TrainingConfig()
    configuration.parse_config(config_file)

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

    print(f"Using device: {device}")
    model = OneChannelResnet(class_labels, pretrained=False)
    model.to(device)

    # criterion = nn.BCEWithLogitsLoss()  # Use this when there is no sigmoid before models output
    criterion = nn.BCELoss()
    print(f"Loss fnc: {criterion}")

    optimizer = optim.Adam(model.parameters())
    print(f"Loss fnc: {optimizer}")

    stats = {"loss": [], "f1": [], "acc": []}

    starting_epoch_idx = 0

    if restart_checkpoint:
        print(f"Loading from checkpoint: {restart_checkpoint}")
        checkpoint = torch.load(restart_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        stats = checkpoint["stats"]
        starting_epoch_idx = checkpoint["epoch"] + 1

    for epoch_idx in range(starting_epoch_idx, starting_epoch_idx + epochs):
        print(f"Epoch: {epoch_idx} - STARTING")
        trn_l, trn_a, trn_f1 = fit_model(epoch_idx, model, train_dataloader, optimizer, criterion, device=device)
        # val_l, val_a = fit_model(i, model, valid_dataloader, phase = 'validation')
        stats["loss"].append(trn_l)
        stats["acc"].append(trn_a)
        stats["f1"].append(trn_f1)


        print(f"Epoch {epoch_idx} average: loss: {trn_l}, acc: {trn_a}, f1: {trn_f1}")

        if epoch_idx % 3 == 0:  # to avoid disk overload save each xth epoch.
            to_save = {
                "config": configuration.config,
                "epoch": epoch_idx,
                "stats": stats,
                "state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() #,
                # 'conf_matrix': conf_matrix
            }
            torch.save(to_save, f"{chpt_folder}/{experiment_name}/checkpoint_epoch{epoch_idx}.pth")


if __name__ == "__main__":
    run_training("configs/training_resnet_double_notes_v0.4.json")

# END
