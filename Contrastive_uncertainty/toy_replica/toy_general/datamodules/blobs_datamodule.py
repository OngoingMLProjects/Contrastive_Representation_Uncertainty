import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,  Dataset,Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

import os
from scipy.io import loadmat
from PIL import Image

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl

import sklearn.datasets
import numpy as np
from math import ceil, floor

from Contrastive_uncertainty.toy_replica.toy_general.datamodules.toy_transforms import CustomTensorDataset



class BlobsDataModule(LightningDataModule): # Data module for Two Moons dataset
    def __init__(self,data_dir: str = None, batch_size=32, seed =42, centers = 10, train_transforms=None, test_transforms=None, multi_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 8 
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.multi_transforms = multi_transforms
        self.centers = centers  # Number of different classes
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
    
    @property
    def num_classes(self):
        """
        Return:
            classes
        """
        return self.centers
    
    
    @property
    def num_channels(self):
        """
        Return:
            0
        """
        return 0
     
    def setup(self, stage=None):
        '''called one each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            self.train_data, self.train_labels = sklearn.datasets.make_blobs(n_samples=1000, centers=self.centers)
            #self.train_data, self.train_labels = sklearn.datasets.make_blobs(n_samples=10000, centers=self.centers)
            
            self.mean, self.std = np.mean(self.train_data,axis = 0), np.std(self.train_data,axis = 0) # calculate the mean and std along a particular dimension

            self.train_data = (self.train_data - self.mean)/self.std #  Normalise the data

            self.val_data, self.val_labels = sklearn.datasets.make_blobs(n_samples=600, centers=self.centers)
            self.val_data = (self.val_data - self.mean)/self.std

        if stage == 'test' or stage is None:
            self.test_data, self.test_labels = sklearn.datasets.make_blobs(n_samples=600, centers=self.centers)
            self.test_data = (self.test_data - self.mean)/self.std

        self.train_dataset = CustomTensorDataset((torch.from_numpy(self.train_data).float(), torch.from_numpy(self.train_labels)),transform = self.train_transforms)
        self.val_train_dataset = CustomTensorDataset((torch.from_numpy(self.val_data).float(), torch.from_numpy(self.val_labels)),transform = self.train_transforms)
        self.val_test_dataset = CustomTensorDataset((torch.from_numpy(self.val_data).float(), torch.from_numpy(self.val_labels)),transform = self.test_transforms)
        
        self.test_dataset = CustomTensorDataset((torch.from_numpy(self.test_data).float(), torch.from_numpy(self.test_labels)),transform = self.test_transforms)

        self.idx2class = {i:f'Class {i}'for i in range(self.centers)}  

    def train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True,num_workers = 8)
        
        return train_loader
    
    def deterministic_train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last = True,num_workers = 8)
        
        return train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        
        val_train_loader = DataLoader(self.val_train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,num_workers = 8) # Batch size is entire validataion set
        val_test_loader = DataLoader(self.val_test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,num_workers = 8)

        return [val_train_loader, val_test_loader]

    def test_dataloader(self):
        '''returns test dataloader'''
        
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)  # Batch size is entire test set
        return test_loader

