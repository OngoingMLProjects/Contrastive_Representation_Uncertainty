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
    def __init__(self,ID_Datamodule, OOD_Datamodule, data_dir:str = None, batch_size =32):
        super().__init__()
        
        self.batch_size = batch_size
        self.ID_Datamodule = ID_Datamodule
        # Remove the coarse labels

        #self.ID_Datamodule.DATASET_with_indices = dataset_with_indices(ID_Datamodule.DATASET)
        #self.ID_Datamodule.setup()
        # Train and test transforms are defined in the datamodule dict 
        #test_transforms = self.ID_Datamodule.test_transforms

        # Update the OOD transforms with the transforms of the ID datamodule
        self.OOD_Datamodule = OOD_Datamodule
        
        self.OOD_Datamodule.train_transforms = self.ID_Datamodule.train_transforms
        self.OOD_Datamodule.test_transforms = self.ID_Datamodule.test_transforms
        
        # Resets the OOD datamodules with the specific transforms of interest required
        self.OOD_Datamodule.setup()

        self.seed = seed

    @property
    def num_classes(self):
        """
        Return:
            classes
        """
        return self.ID_Datamodule.num_classes

    def setup(self):
        pass

    def train_dataloader(self):
        '''returns training dataloader'''
        
        loader_ID = DataLoader(self.ID_Datamodule.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True,num_workers = 8)
        loader_OOD = DataLoader(self.OOD_Datamodule.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last = True,num_workers = 8)

        train_loader = {"ID":loader_ID, "OOD": loader_OOD}
        return train_loader
    
    def deterministic_train_dataloader(self):
        '''returns training dataloader'''

        loader_ID = DataLoader(self.ID_Datamodule.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last = True,num_workers = 8)
        loader_OOD = DataLoader(self.OOD_Datamodule.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last = True,num_workers = 8)

        deterministic_train_loader = {"ID":loader_ID, "OOD": loader_OOD}
        return deterministic_train_loader

    def val_dataloader(self):
        '''returns validation dataloader'''

        ID_val_train_loader = DataLoader(
            self.ID_Datamodule.val_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        OOD_val_train_loader = DataLoader(
            self.OOD_Datamodule.val_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_train_loader = {"ID":ID_val_train_loader,"OOD":OOD_val_train_loader}

        ID_val_test_loader = DataLoader(
            self.ID_Datamodule.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        OOD_val_test_loader = DataLoader(
            self.OOD_Datamodule.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_test_loader = {"ID":ID_val_test_loader,"OOD":OOD_val_test_loader}

        return [val_train_loader, val_test_loader]

    def test_dataloader(self):
        '''returns test dataloader'''
        loader_ID = DataLoader(self.ID_Datamodule.test_dataset,batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)
        loader_OOD = DataLoader(self.OOD_Datamodule.test_dataset,batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)
        test_loader = {"ID":loader_ID, "OOD": loader_OOD}
        return test_loader
    
'''
ID_datamodule = MNISTDataModule()
OOD_datamodule = FashionMNISTDataModule()
ID_datamodule.setup()
OOD_datamodule.setup()
OE_module= OEDatamodule(ID_datamodule,OOD_datamodule)
train_loader = OE_module.train_dataloader()
test_loader = OE_module.test_dataloader()
'''