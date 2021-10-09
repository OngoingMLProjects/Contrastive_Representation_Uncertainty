import os
from typing import Optional, Sequence
import numpy as np
from pytorch_lightning.core import datamodule
import torch
import torchvision
from torchvision.datasets import caltech
from torchvision.datasets.caltech import Caltech101


#from datalad.api import download_url
#import patoolib

from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader, random_split

from warnings import warn
import copy
import pickle


from torchvision import transforms as transform_lib
from torchvision.datasets import Caltech101
from torchvision.transforms import transforms

from Contrastive_uncertainty.general.datamodules.dataset_normalizations import imagenet_normalization
from Contrastive_uncertainty.general.datamodules.datamodule_transforms import dataset_with_indices, UpdatedCustomTensorDataset

# http://places2.csail.mit.edu/download.html where I downloaded the dataset for the case of places 365
# based on https://pretagteam.com/question/pytorch-lightning-get-models-output-on-full-train-data-during-training
class ImageNetDataModule(LightningDataModule):

    name = 'imagenet'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 500,
            num_workers: int = 16,
            batch_size: int = 256,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
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
            1000
        """
        return 1000
    
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
            32
        """
        return 32
        
    
    def prepare_data(self):
        """

        """
        pass
            
    def setup(self):
        # https://discuss.pytorch.org/t/how-can-i-make-npz-dataloader/94938
        # val_data = 'ImageNet/imagenet32_val/val_data'    
        # val_dataset = np.load(val_data)
        
        train_datafolder = 'ImageNet/Imagenet32_train'
        collated_X_data,collated_Y_data = [], [] 
        val_datafolder = 'ImageNet/Imagenet32_val/val_data' 
        
        for i in range(10):
            X_train,Y_train = self.load_databatch(train_datafolder,i+1)    
            collated_X_data.append(X_train), collated_Y_data.append(Y_train)
        
        X_train = torch.cat(collated_X_data)
        Y_train = torch.cat(collated_Y_data)


        X_val, Y_val = self.load_databatch(val_datafolder,idx=0)
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        
        train_dataset = UpdatedCustomTensorDataset((X_train, Y_train),transform = train_transforms)
        val_dataset = UpdatedCustomTensorDataset((X_val, Y_val),transform = test_transforms)
        
        # Use a subset of 1/10 image net for training purposes
        _ , self.train_dataset, self.val_train_dataset = random_split(train_dataset, [1_146_167,130_000,5000],generator=torch.Generator().manual_seed(self.seed)
        )
        
        _, self.val_test_dataset, self.test_dataset = random_split(val_dataset, [35_000,5000, 10000],generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Steps to include:
        # Difficulty cloning the dataset to use different transforms. A few approaches which can be used is determinstically obtaining a subset of the data each time https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
        # Also could look at separating the data several times 
        # Within the custom dataset transform, make it so that it transofmrs to numpy array and PIL image within the transform https://github.com/pytorch/vision/blob/972ca657416c5896ba6e0f5fe7c0d0f3e99e1087/torchvision/datasets/mnist.py#L501
        # Calculate normalisation
        #
        
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def load_databatch(self,datafolder,idx,img_size=32):
        train_data_file = os.path.join(datafolder,'train_data_batch_')
        val_datafile = datafolder
        
        if idx ==0:
            d = self.unpickle(val_datafile)
        else:
            d = self.unpickle(train_data_file + str(idx))
        
        x = d['data']
        y = d['labels']

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]

        data_size = x.shape[0]
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)

        X_data = x[0:data_size, :, :, :]
        Y_data = y[0:data_size]
        return torch.tensor(X_data), torch.tensor(Y_data)


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
        imagenet_transforms = transform_lib.Compose([
            transforms.Resize(size = (32,32)),
            transform_lib.ToTensor(),
            imagenet_normalization()
        ])
        return imagenet_transforms

'''
datamodule = ImageNetDataModule()
datamodule.setup()


test_loader = datamodule.test_dataloader()
train_loader = datamodule.deterministic_train_dataloader()

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
'''
'''
datamodule = Caltech101DataModule()
#datamodule.prepare_data()
datamodule.setup()

test_loader = datamodule.test_dataloader()
train_loader = datamodule.deterministic_train_dataloader()
for i,k in zip(train_loader,test_loader):
'''