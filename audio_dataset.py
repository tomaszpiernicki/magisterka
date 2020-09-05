import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

# import config

class AudioDataset(Dataset):

    def __init__(self, feat_folder, class_labels, folds):
        self.feat_folder = feat_folder
        self.labels_dict = class_labels
        self.numberOfFolds = folds
        self.load_fold(0)

    def load_fold(self, fold_idx):
        fold_idx = fold_idx % self.numberOfFolds  # reuse folds when there are no new ones
        feat_files = glob.glob(self.feat_folder + f'mbe_*_fold{fold_idx}_*.npz')
        self.feat_dict = dict(enumerate(feat_files))

    def load_data(self, feat_file):
        dmp = np.load(feat_file, allow_pickle=True)
        _X_train, _Y_train = dmp['arr_0'], dmp['arr_1']
        return _X_train, _Y_train

    def __getitem__(self, index):
        file = self.feat_dict[index]
        data, label = self.load_data(file)
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data, label

    def __len__(self):
        return len(self.feat_dict.keys())

# END
