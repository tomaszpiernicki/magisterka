import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

import config

import glob

import re



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


def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)


def find_last_epoch():
    chpts_files = glob.glob(config.chpt_folder + '*.pth')
    if len(chpts_files) == 0:
        return 0
    epochs = [int(re.findall('[0-9]+', file)[-1]) for file in chpts_files]
    return max(epochs)

#
# dataset = AudioDataset()
#
# train_len = int(len(dataset) * 0.8)
# valid_len = len(dataset) - train_len
#
# train_dataset, valid_dataset = torch.utils.data.dataset10 876.random_split(dataset, [train_len, valid_len])
#
# batch_size = 2
#
# train_dl = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0
# )
#
# valid_dl = DataLoader(
#     valid_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=0
# )
#
# print(len(dataset.labels_dict))
#
# # criterion = nn.BCELoss()
# criterion = nn.MultiLabelSoftMarginLoss()
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"DEVICE: {DEVICE}")
#
# model = Net(40, len(dataset.labels_dict))
# model.to(DEVICE)
#
# optimizer = torch.optim.Adam(
#     get_trainable(model.parameters()),
#     lr=1e-3,
#     betas=(0.9, 0.999),
#     eps=1e-08,
#     weight_decay=0
# )
#
# last_epoch = find_last_epoch()
#
# if last_epoch > 0:
#     checkpoint = torch.load(f"{config.chpt_folder}checkpoint_{last_epoch}.pth")
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#
# N_EPOCHS = 500
# for epoch in range(last_epoch, N_EPOCHS):
#
#     epoch_info = {}
#     #
#     # confusion_matrix = {
#     #     'y_Actual': np.asarray([]),
#     #     'y_Predicted': np.asarr ay([])
#     # }
#
#     # Train mode
#     model.train()
#
#     total_loss, n_correct, n_samples = 0.0, 0, 0
#
#     for batch_i, (X, y) in enumerate(train_dl):
#         X, y = X.float().to(DEVICE), y.float().to(DEVICE)
#
#         optimizer.zero_grad()
#
#         y_ = model.forward(X)
#
#         loss = criterion(y_, y)
#         loss.backward()
#         optimizer.step()
#
#         # Statistics
#         print(
#             f"Epoch {epoch + 1}/{N_EPOCHS} |"
#             f"  batch: {batch_i} |"
#             f"  batch loss:   {loss.item():0.3f}"
#         )
#
#         y_label_ = y_.round()
#
#         # _, y_label_ = torch.max(y_, 1)
#
#         n_correct += (y_label_ == y).sum().item()
#
#         statistics = metrics.compute_scores(y_label_, y)
#
#         total_loss += loss.item() * X.shape[0]
#         n_samples += X.shape[0]
#
#     print(
#         f"Epoch {epoch + 1}/{N_EPOCHS} |"
#         f"  valid %: {total_loss / n_samples} |"
#         f"  valid acc:  {n_correct / n_samples * 100:9.3f}%"
#     )
#
#     model.eval()  # important
#
#     total_loss, n_correct, n_samples = 0.0, 0, 0
#     with torch.no_grad():  # IMPORTANT
#         for X, y in valid_dl:
#             X, y = X.float().to(DEVICE), y.float().to(DEVICE)
#
#             y_ = model(X)
#
#             # Statistics
#             y_label_ = y_.round()
#
#             n_correct += (y_label_ == y).sum().item()
#             loss = criterion(y_, y)
#             total_loss += loss.item() * X.shape[0]
#             n_samples += X.shape[0]
#
#             # confusion_matrix['y_Actual'] = np.append(confusion_matrix['y_Actual'], y.cpu().numpy())
#             # confusion_matrix['y_Predicted'] = np.append(confusion_matrix['y_Predicted'], y_label_.cpu().numpy())
#
#     train_loss = total_loss / n_samples
#     train_acc = n_correct / n_samples * 100
#
#     print(
#         f"Epoch {epoch + 1}/{N_EPOCHS} |"
#         f"  train : {total_loss / n_samples:9.3f} |"
#         f"  valid acc:  {train_acc:9.3f}%"
#     )
#
#     epoch_info["train_loss"] = train_loss
#     epoch_info["train_acc"] = train_acc
#     epoch_info["index"] = epoch + 1
#
#     if epoch % 5 == 0:
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'meta': epoch_info  # ,
#             # 'conf_matrix' : nfusion_matrix
#         }, f"{config.chpt_folder}checkpoint_{epoch + 1}.pth")
#
#     with open(f"{config.chpt_folder}epoch_info.json", 'a+') as fp:
#         json.dump(epoch_info, fp)
#
#
# def train_epoch(model, n_epochs=1000, starting_epoch=0, device="cuda"):
#     for epoch_idx in range(starting_epoch, n_epochs):
#         model.train()
#
#         for batch_i, (input_, target_) in enumerate(train_dl):
#             input_, target_ = input_.float().to(DEVICE), target_.float().to(DEVICE)
#
#             optimizer.zero_grad()
#
#             output_ = model.forward(input_)
#
#             loss_ = criterion(output_, target_)
#             loss_.backward()
#
#             optimizer.step()
#
#
# def evaluate():
#     pass
