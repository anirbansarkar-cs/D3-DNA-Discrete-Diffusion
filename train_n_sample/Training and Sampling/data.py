import re
# from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
import h5py, os
# from datasets import Dataset

'''
Please follow codes from "Dirichlet-flow-matching" (https://github.com/HannesStark/dirichlet-flow-matching) and "Dirichlet diffusion score model" (https://github.com/jzhoulab/ddsm)
for setting up the code to train for Promoter dataset and run import by uncommenting below line.
'''
# from promoter_dataset import PromoterDataset

from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data



def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    # Deepstarr
    filepath = os.path.join('DeepSTARR_data.h5')
    data = h5py.File(filepath, 'r')
    X_train = torch.tensor(np.array(data['X_train']))
    y_train = torch.tensor(np.array(data['Y_train']))
    X_train = torch.argmax(X_train, dim=1)
    X_valid = torch.tensor(np.array(data['X_valid']))
    y_valid = torch.tensor(np.array(data['Y_valid']))
    X_valid = torch.argmax(X_valid, dim=1)
    train_set = TensorDataset(X_train, y_train)
    valid_set = TensorDataset(X_valid, y_valid)

    #MPRA
    # filepath = os.path.join('mpra_data.h5')
    # dataset = h5py.File(filepath, 'r')
    # x_train = torch.tensor(np.array(dataset['x_train']).astype(np.float32)).permute(0,2,1)
    # x_train = torch.argmax(x_train, dim=1)
    # y_train = torch.tensor(np.array(dataset['y_train']).astype(np.float32))
    # x_valid = torch.tensor(np.array(dataset['x_valid']).astype(np.float32)).permute(0,2,1)
    # x_valid = torch.argmax(x_valid, dim=1)
    # y_valid = torch.tensor(np.array(dataset['y_valid']).astype(np.float32))
    # x_test = np.array(dataset['x_test']).astype(np.float32)
    # y_test = np.array(dataset['y_test']).astype(np.float32)
    #
    # train_set = TensorDataset(x_train, y_train)
    # valid_set = TensorDataset(x_valid, y_valid)

    # Promoter
    # train_set = torch.tensor(np.array(PromoterDataset(n_tsses=100000, rand_offset=10, split='train')))
    # valid_set = torch.tensor(np.array(PromoterDataset(n_tsses=100000, rand_offset=0, split='test')))

    print (len(train_set), len(valid_set))

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))


    return train_loader, valid_loader

