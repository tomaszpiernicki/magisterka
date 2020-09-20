import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

from old import config


class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),  # 3, 256, 256
            nn.MaxPool2d(2),  # op: 16, 127, 127
            nn.ReLU(),  # op: 64, 127, 127
        )
        self.ConvLayer2 = nn.Sequential(nn.Conv2d(64, 128, 3),  # 64, 127, 127
                                        nn.MaxPool2d(2),  # op: 128, 63, 63
                                        nn.ReLU()  # op: 128, 63, 63
                                        )
        self.ConvLayer3 = nn.Sequential(nn.Conv2d(128, 256, 3),  # 128, 63, 63
                                        nn.MaxPool2d(2),  # op: 256, 30, 30
                                        nn.ReLU()  # op: 256, 30, 30
                                        )
        self.ConvLayer4 = nn.Sequential(nn.Conv2d(256, 512, 3),  # 256, 30, 30
                                        nn.MaxPool2d(2),  # op: 512, 14, 14
                                        nn.ReLU(),  # op: 512, 14, 14
                                        nn.Dropout(0.2))
        self.Linear1 = nn.Linear(512 * 14 * 14, 1024)
        self.Linear2 = nn.Linear(1024, 256)
        self.Linear3 = nn.Linear(256, len(config.class_labels.keys()))

    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return F.sigmoid(x)


class OneChannelResnet(nn.Module):
    def __init__(self, class_labels, pretrained = False):
        super(OneChannelResnet, self).__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(512 * self.model.layer1[0].expansion, len(class_labels.keys()))

    def forward(self, x):
        x = self.model(x)
        return F.sigmoid(x)


class Net(nn.Module):
    # def __init__(self, n_concat, n_freq, n_out):
    def __init__(self, n_in, n_out):
        super(Net, self).__init__()
        # n_in = n_concat * n_freq
        n_hid = 200

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_out)

    def forward(self, x):
        drop_p = 0.2
        # x1 = x.view(len(x), -1)
        x1 = self.fc1(x)
        x2 = F.dropout(F.relu(x1), p=drop_p, training=True)
        x2 = self.fc2(x2)
        x3 = F.dropout(F.relu(x2), p=drop_p, training=True)
        x3 = self.fc3(x3)
        x4 = F.dropout(F.relu(x3), p=drop_p, training=True)
        x4 = self.fc4(x4)
        x5 = F.sigmoid(x4)
        return x5
