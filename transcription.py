from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
import metrics
import utils
from IPython import embed

import torch
from torch import nn

import config

import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchsummary

import torchvision
from torchvision import models
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock

import torch.onnx

import pandas as pd

import seaborn as sn

import numpy as np

import matplotlib.pyplot as plt

import pdb

import os


logger = logging.getLogger(__name__)

def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp = np.load(feat_file_fold, allow_pickle=True)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    return _X_train, _Y_train, _X_test, _Y_test

def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test

plot.switch_backend('agg')
sys.setrecursionlimit(10000)

# CRNN model definition
cnn_nb_filt = 128            # CNN filter size
cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
rnn_nb = [32, 32]           # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
dropout_rate = 0.5          # Dropout after each layer

seq_len = 256       # Frame sequence length. Input to the CRNN.

print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
    cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

conv_channels = [[ config.nb_ch, cnn_nb_filt],
                  [cnn_nb_filt, cnn_nb_filt],
                  [cnn_nb_filt, cnn_nb_filt]]

pool_size = [[1,5], [1, 2], [1, 2]]

conv_kernel = [3, 3]

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class ConvLayer(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, padding = 1, dropout_rate=0.5):
        super().__init__()

        # Shape: time-samples, Channels, Time-frames, Mel-features
        #          21           1       256     40

        # Convolution across time-frames x mel-features
        self.conv = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class TransctiptionModel(nn.Module):
    def __init__(self, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):

        self.conv1 = ConvLayer(conv_channels[0], [3, 3], pool_size[0])
        self.conv2 = ConvLayer(conv_channels[1], [3, 3], pool_size[1])
        self.conv3 = ConvLayer(conv_channels[2], [3, 3], pool_size[2])

        self.gru = nn.GRU(256, _rnn_nb[0], num_layers=len(_rnn_nb), dropout=dropout_rate, batch_first=True, bidirectional=True )

        dense1 = nn.Linear(256, 32)
        self.td1 = TimeDistributed(dense1, batch_first=True)

        dense2 = nn.Linear(256, 100)
        self.td2 = TimeDistributed(dense2, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x.permute(0, 2, 1, 3)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)

        x = self.gru(x)

        x = self.td1(x)
        x = self.td2(x)

        return self.sigmoid(x)





X, Y, X_test, Y_test = load_data(config.feat_folder, config.is_mono, 1)
X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, config.nb_ch)


