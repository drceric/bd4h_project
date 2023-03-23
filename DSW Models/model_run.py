import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from model_synthetic_package import LSTMModel
from train_synthetic_package import trainInitIPTW

# dataset meta data
n_X_features = 100
n_X_static_features = 5
n_X_t_types = 1
n_classes = 1


def get_dim():
    return n_X_features, n_X_static_features, n_X_t_types, n_classes


class SyntheticDataset(data.Dataset):
    def __init__(self, list_IDs, obs_w, treatment):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.obs_w = obs_w
        self.treatment = treatment


    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        ID = self.list_IDs[index]

        # Load labels
        label = np.load(data_dir + '{}.y.npy'.format(ID))

        # Load data
        X_demographic = np.load(data_dir + '{}.static.npy'.format(ID))
        X_all = np.load(data_dir + '{}.x.npy'.format(ID))
        X_treatment_res = np.load(data_dir + '{}.a.npy'.format(ID))

        X = torch.from_numpy(X_all.astype(np.float32))
        X_demo = torch.from_numpy(X_demographic.astype(np.float32))
        X_treatment = torch.from_numpy(X_treatment_res.astype(np.float32))
        y = torch.from_numpy(label.astype(np.float32))

        return X, X_demo, X_treatment, y

#Default Parameters
treatment_option = 'vaso'
observation_window = 12
epochs = 30
batch_size = 128
lr = .001
weight_decay = .00001
l1 = .00001
resume = ''.format(treatment_option)
cuda_device = 1

gamma_h=(.1,.3,.5,.7)
HIDDEN_SIZE = 32
CUDA = False

os.makedirs(r'model_checkpoints', exist_ok=True)
for gamma in gamma_h:
    data_dir = 'data_synthetic/data_syn_{}/'.format(gamma)
    save_model = 'model_checkpoints/mimic-6-7-{}.pt'.format(gamma)
    train_test_split = np.loadtxt('data_synthetic/data_syn_{}/train_test_split.csv'.format(gamma), delimiter=',',
                                  dtype=int)
    train_iids = np.where(train_test_split == 1)[0]
    val_iids = np.where(train_test_split == 2)[0]
    test_iids = np.where(train_test_split == 0)[0]
    train_dataset = SyntheticDataset(train_iids, 12, treatment_option)
    val_dataset = SyntheticDataset(val_iids, 12, treatment_option)
    test_dataset = SyntheticDataset(test_iids, 12, treatment_option)
    train_loader = torch.utils.data.DataLoader(train_dataset, 128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=True)
    n_X_features, n_X_static_features, n_X_fr_types, n_classes = get_dim()
    if ''.format(treatment_option):
        if os.path.isfile(''.format(treatment_option)):
            print("=> loading checkpoint '{}'".format(''.format(treatment_option)))

            model = torch.load(''.format(treatment_option))
            model = model.cuda()

            print("=> loaded checkpoint '{}'"
                  .format(''.format(treatment_option)))

        else:
            print("=> no checkpoint found at '{}'".format(''.format(treatment_option)))
    else:

        attn_model = 'concat2'
        n_Z_confounders = HIDDEN_SIZE

        model = LSTMModel(n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                          attn_model, n_classes, 12,
                          128, hidden_size=HIDDEN_SIZE)

    adam_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model = trainInitIPTW(train_loader, val_loader, test_loader,
                          model, epochs=30,
                          criterion=F.binary_cross_entropy_with_logits, optimizer=adam_optimizer,
                          l1_reg_coef=1e-5,
                          use_cuda=False,
                          save_model=save_model)