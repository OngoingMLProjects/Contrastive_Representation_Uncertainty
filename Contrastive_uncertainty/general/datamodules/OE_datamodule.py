from typing import List
from pytorch_lightning.core import datamodule
from Contrastive_uncertainty.general.train.train_general import train
from pytorch_lightning.utilities import seed
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,  Dataset,Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import copy

import os
from scipy.io import loadmat
from PIL import Image

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl

import sklearn.datasets
import numpy as np
from math import ceil, floor

from Contrastive_uncertainty.general.datamodules.mnist_datamodule import MNISTDataModule
from Contrastive_uncertainty.general.datamodules.fashionmnist_datamodule import FashionMNISTDataModule
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2457
# https://github.com/PyTorchLightning/pytorch-lightning/pull/1959
# https://github.com/PyTorchLightning/pytorch-lightning/pull/1959#issuecomment-655613076



# Outlier exposure datamodule
class OEDatamodule(LightningDataModule):
    def __init__(self,ID_Datamodule, OOD_Datamodule, *args,
            **kwargs):
        super().__init__(*args,**kwargs)
        
        
        self.ID_Datamodule = ID_Datamodule
        self.batch_size = self.ID_Datamodule.batch_size
        self.num_workers = self.ID_Datamodule.num_workers
        self.data_dir = self.ID_Datamodule.data_dir
        self.seed = self.ID_Datamodule.seed

        # Remove the coarse labels

        #self.ID_Datamodule.DATASET_with_indices = dataset_with_indices(ID_Datamodule.DATASET)
        #self.ID_Datamodule.setup()
        # Train and test transforms are defined in the datamodule dict 
        #test_transforms = self.ID_Datamodule.test_transforms

        # Update the OOD transforms with the transforms of the ID datamodule
        self.OOD_Datamodule = OOD_Datamodule
        
        self.OOD_Datamodule.train_transforms = self.ID_Datamodule.train_transforms
        self.OOD_Datamodule.test_transforms = self.ID_Datamodule.test_transforms
        self.OOD_Datamodule.setup()
        # Resets the OOD datamodules with the specific transforms of interest required
        

        

    @property
    def num_classes(self):
        """
        Return:
            classes
        """
        return self.ID_Datamodule.num_classes
    
    @property
    def num_channels(self):
        """
        Return:
            classes
        """
        return self.ID_Datamodule.num_channels

    def prepare_data(self):
        pass 

    def setup(self,stage=None): #  Need to use stagename for some reason
        
        train_datasets = [self.ID_Datamodule.train_dataset,self.OOD_Datamodule.train_dataset]
        val_train_datasets = [self.ID_Datamodule.val_train_dataset,self.OOD_Datamodule.val_train_dataset]
        val_test_datasets = [self.ID_Datamodule.val_test_dataset,self.OOD_Datamodule.val_test_dataset]
        test_datasets = [self.ID_Datamodule.test_dataset,self.OOD_Datamodule.test_dataset]

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_train_dataset = ConcatDataset(val_train_datasets)
        self.val_test_dataset = ConcatDataset(val_test_datasets)
        self.test_dataset = ConcatDataset(test_datasets)
        '''
        self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        self.val_train_dataset = torch.utils.data.ConcatDataset(val_train_datasets)
        self.val_test_dataset = torch.utils.data.ConcatDataset(val_test_datasets)
        self.test_dataset = torch.utils.data.ConcatDataset(test_datasets)
        '''

    def train_dataloader(self):
        '''returns training dataloader'''
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True, num_workers = 8)     

        return train_loader
    
    def deterministic_train_dataloader(self):
        import ipdb; ipdb.set_trace()
        '''returns training dataloader'''
        ''
        deterministic_train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last = True, num_workers = 8)

        return deterministic_train_loader


    def val_dataloader(self):
        '''returns validation dataloader'''

        val_train_loader = DataLoader(
            self.val_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_test_loader = DataLoader(
            self.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return [val_train_loader, val_test_loader]

    def test_dataloader(self):
        '''returns test dataloader'''
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True, num_workers = 8)
        return test_loader

'''
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
'''
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])

        return tuple(result)

    def __len__(self):
        return min(len(d) for d in self.datasets)



ID_datamodule = MNISTDataModule()
OOD_datamodule = FashionMNISTDataModule()
ID_datamodule.setup()
OOD_datamodule.setup()
OE_module= OEDatamodule(ID_datamodule,OOD_datamodule)
OE_module.setup()

#train_loader = OE_module.train_dataloader()
#test_loader = OE_module.test_dataloader()

loader = OE_module.deterministic_train_dataloader()
import ipdb; ipdb.set_trace()
