# TODO: LEGACY SCRIPT - to be removed

from functools import partial
from logging import Logger
from time import time

import torch

import seaborn
from torchvision import models

import config

from maps_dataset import Net, get_trainable, AudioDataset

import torch.nn as nn
from torch.utils.data import DataLoader

from fastai.vision import *

def train(model, optimizer, loss_f, train_dl, device, n_epochs=1000, starting_epoch=0):
    print(f"Starting training with parameters: \n model: {model} \n  optimizer: {optimizer}, \n loss_f: {loss_f}, \n device: {device} ")

    for epoch_idx in range(starting_epoch, n_epochs):

        print(f"Starting epoch: {epoch_idx}")

        epoch_info = {}
        epoch_info["epoch_idx"] = epoch_idx

        model.train()

        epoch_loss = 0.0
        n_samples = 0

        for batch_i, (input_, target_) in enumerate(train_dl):

            start_time_ = time()
            input_, target_ = input_.float().to(device), target_.float().to(device)

            optimizer.zero_grad()

            output_ = model.forward(input_)

            output_labels = output_.round()

            loss_ = loss_f(output_labels, target_)
            loss_.backward()

            optimizer.step()

            epoch_loss += loss_.item() * input_.shape[0]
            time_taken_ = time() - start_time_

            n_samples += input_.shape[0]

            heatmap = seaborn.heatmap(target_.detach()[0].cpu()).get_figure()
            heatmap.savefig("heatmap_target.png")

            # heatmap = seaborn.heatmap(output_labels.detach()[0].cpu()).get_figure()
            # heatmap.savefig("heatmap_output_labels.png")

            print(f"\t loss: {loss_}, time: {time_taken_} ")

        epoch_info["loss"] = epoch_loss / n_samples

        print(f"Epoch loss: {epoch_info['loss']}")

        if epoch_idx % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'meta': epoch_info
            }, f"{config.chpt_folder}checkpoint_{epoch_idx + 1}.pth")


            # TODO: Check for buga
            # TODO: collect valuable data.
            # TODO: save epoch_info


from pprint import pprint
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)


def fit_model(epochs, model, dataloader, phase='training', volatile=False):
    pprint("Epoch: {}".format(epochs))
    if phase == 'training':
        model.train()

    if phase == 'validataion':
        model.eval()
        volatile = True

    running_loss = []
    running_acc = []
    b = 0
    for i, data in enumerate(dataloader):

        inputs, target = data['image'].cuda(), data['label'].float().cuda()

        inputs, target = Variable(inputs), Variable(target)

        if phase == 'training':
            optimizer.zero_grad()

        ops = model(inputs)
        acc_ = []
        for i, d in enumerate(ops, 0): acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d))
        acc_.append(acc)
        loss = criterion(ops, target)

        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        b += 1

        if phase == 'training':
            loss.backward()

            optimizer.step()

    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()

    pprint("{} loss is {} ".format(phase, total_batch_loss))
    pprint("{} accuracy is {} ".format(phase, total_batch_acc))

    return total_batch_loss, total_batch_ac


def train_using_fast():
    pass


class OneChannelResnet(nn.Module):
    def __init__(self, pretrained = False):
        super(OneChannelResnet, self).__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(512 * self.model.layer1[0].expansion, 1)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    logger = Logger("Training")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    data_set = AudioDataset()
    data_set_len = len(data_set)
    train_test_ratio = 0.1

    train_set, eval_set = torch.utils.data.random_split(data_set, [int(data_set_len * train_test_ratio),
                                                                  int(data_set_len * (1 - train_test_ratio))])

    batch_size = 2

    train_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    eval_dl = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data = ImageDataBunch(train_dl, eval_dl)
    data.c = len(config.__class_labels.keys())
    # # Model
    # model = Net(40, len(train_ds.labels_dict))
    # model.to(device)

    model = OneChannelResnet

    # Define optimizer
    # optimizer = torch.optim.Adam(
    #     get_trainable(model.parameters()),
    #     lr=1e-3,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=0
    # )

    # define loss funtion
    # loss_f = nn.MultiLabelSoftMarginLoss()
    # loss_f = nn.BCEWithLogitsLoss()
    # loss_f = nn.BCELoss()

    # train(model, optimizer, loss_f, train_dl, device)

    acc_02 = partial(accuracy_thresh, thresh=0.2)
    f_score = partial(fbeta, thresh=0.2)


    learn = cnn_learner(data, model, metrics=[acc_02, f_score])

    learn.lr_find()

    learn.recorder.plot()