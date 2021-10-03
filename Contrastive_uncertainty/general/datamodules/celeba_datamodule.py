import os
from typing import Optional, Sequence
import numpy as np
from pytorch_lightning.core import datamodule
import torch
import torchvision

from torchvision.datasets import CelebA


#from datalad.api import download_url
#import patoolib

from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader, random_split

from warnings import warn
import copy


from torchvision import transforms as transform_lib
from torchvision.transforms import transforms

from Contrastive_uncertainty.general.datamodules.dataset_normalizations import celeba_normalization
from Contrastive_uncertainty.general.datamodules.datamodule_transforms import dataset_with_indices

# based on https://pretagteam.com/question/pytorch-lightning-get-models-output-on-full-train-data-during-training
class CelebADataModule(LightningDataModule):

    name = 'celeba'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 500,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # https://paperswithcode.com/dataset/celeba
        self.dims = (3, 178, 218)
        self.DATASET = CelebA
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 5000 - val_split


    @property
    def total_samples(self):
        """
        Return:
            5000
        """
        return 5000
        
    @property
    def num_classes(self):
        """
        Return:
            1
        """
        return 1
    
    @property
    def num_channels(self):
        """
        Return:
            3
        """
        return 3
    
    @property
    def input_height(self):
        """
        Return:
            178
        """
        return 178
        
    
    def prepare_data(self):
        """
        Saves CelebA files to data_dir
        """
        #torchvision.datasets.CelebA(self.data_dir)
        pass # Using pass as I need to download the data directly as there does seem to be errors in the downloading of the data
        #self.DATASET(self.data_dir, split ='all', download=True,transform=transform_lib.ToTensor())
        #self.DATASET(self.data_dir, split ='test', download=True,transform=transform_lib.ToTensor())
        
    
    def setup(self):
        data_path = 'celeba/'
        Indices_ImageFolder =dataset_with_indices(torchvision.datasets.ImageFolder)

        celeba_dataset = Indices_ImageFolder(
            root=data_path,
        transform=torchvision.transforms.ToTensor())
        
        self.idx2class = {i:f'class {i}' for i in range(max(celeba_dataset.targets)+1)}
        if isinstance(celeba_dataset.targets, list):
            celeba_dataset.targets = torch.Tensor(celeba_dataset.targets).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(celeba_dataset.targets,np.ndarray):
            celeba_dataset.targets = torch.from_numpy(celeba_dataset.targets).type(torch.int64)

        # Same validataion and test set size as CIFAR10
        train_dataset, val_dataset, test_dataset = random_split(celeba_dataset, [187599, 5000, 10000],generator=torch.Generator().manual_seed(self.seed)
        )

        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        
        # Need to use deep copy in order to enable the train and test set to have distinct augmentations, otherwise the data augmentation will be postponed each time
        self.train_dataset = copy.deepcopy(train_dataset)
        self.test_dataset = copy.deepcopy(test_dataset)

        self.train_dataset.dataset.transform = train_transforms
        self.test_dataset.dataset.transform = test_transforms

        self.val_train_dataset = copy.deepcopy(val_dataset) 
        self.val_test_dataset = copy.deepcopy(val_dataset)

        self.val_train_dataset.dataset.transform = train_transforms
        self.val_test_dataset.dataset.transform = test_transforms
    
    def train_dataloader(self):
        """
        FashionMNIST train set removes a subset to use for validation
        """

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader
    
    def deterministic_train_dataloader(self): #  Makes it so that the data does not shuffle (Used for the case of the hierarchical approach)
        """
        FashionMNIST train set removes a subset to use for validation
        """

        
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        
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
        """
        FashionMNIST test set uses the test split
        """

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        celeba_transforms = transform_lib.Compose([
            transforms.Resize(size = (178,178)),
            transform_lib.ToTensor(),
            #celeba_normalization()
        ])
        return celeba_transforms

'''
datamodule = CelebADataModule()
datamodule.setup()
test_loader = datamodule.test_dataloader()
train_loader = datamodule.deterministic_train_dataloader()
'''

'''
for i,k in zip(train_loader,test_loader):
    import ipdb; ipdb.set_trace()
'''



'''
mean = 0.
std = 0.
nb_samples = 0.
for data in train_loader:
    batch_samples = data[0].size(0)
    data = data[0].view(batch_samples, data[0].size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
import ipdb; ipdb.set_trace()
'''