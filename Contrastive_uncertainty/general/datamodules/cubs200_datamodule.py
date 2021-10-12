import os
import pandas as pd
import shutil
from typing import Optional, Sequence
import numpy as np
from pytorch_lightning.core import datamodule
import torch
import torchvision


#from datalad.api import download_url
#import patoolib

from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader, random_split

from warnings import warn
import copy


from torchvision import transforms as transform_lib
from torchvision.transforms import transforms

from Contrastive_uncertainty.general.datamodules.dataset_normalizations import cubs200_normalization
from Contrastive_uncertainty.general.datamodules.datamodule_transforms import dataset_with_indices

# Based on this repository  - https://github.com/ecm200/caltech_birds
# https://github.com/TDeVries/cub2011_dataset    Potential alternative
class CUB200DataModule(LightningDataModule):

    name = 'cub200'
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
            100_000
        """
        return 100_0000
        
    @property
    def num_classes(self):
        """
        Return:
            200
        """
        return 200
    
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
        return 32 # 64 in actuality
        
    
    def prepare_data(self):
        '''
        root_dir = 'CUB_200_2011/CUB_200_2011'
        orig_images_folder = 'images_orig'
        new_images_folder = 'images'

        data_dir = os.path.join(root_dir,orig_images_folder)
        new_data_dir = os.path.join(root_dir,new_images_folder)
        image_fnames = pd.read_csv(filepath_or_buffer=os.path.join(root_dir,'images.txt'), 
                          header=None, 
                          delimiter=' ', 
                          names=['Img ID', 'file path'])

        image_fnames['is training image?'] = pd.read_csv(filepath_or_buffer=os.path.join(root_dir,'train_test_split.txt'), 
                                                 header=None, delimiter=' ', 
                                                 names=['Img ID','is training image?'])['is training image?']
        os.makedirs(os.path.join(new_data_dir,'train'), exist_ok=True)
        os.makedirs(os.path.join(new_data_dir,'test'), exist_ok=True)

        for i_image, image_fname in enumerate(image_fnames['file path']):
            if image_fnames['is training image?'].iloc[i_image]:
                new_dir = os.path.join(new_data_dir,'train',image_fname.split('/')[0])
                os.makedirs(new_dir, exist_ok=True)
                shutil.copy(src=os.path.join(data_dir,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
                print(i_image, ':: Image is in training set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
                print('Image:: ', image_fname)
                print('Destination:: ', new_dir)
            else:
                new_dir = os.path.join(new_data_dir,'test',image_fname.split('/')[0])
                os.makedirs(new_dir, exist_ok=True)
                shutil.copy(src=os.path.join(data_dir,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
                print(i_image, ':: Image is in testing set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
                print('Source Image:: ', image_fname)
                print('Destination:: ', new_dir)
        '''
        pass
        
    def setup(self):
        data_dir = 'CUB_200_2011/CUB_200_2011/images'
        train_data_dir = os.path.join(data_dir,'train')
        test_data_dir = os.path.join(data_dir,'test')
        Indices_ImageFolder = dataset_with_indices(torchvision.datasets.ImageFolder)
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        cubs200_train_dataset = Indices_ImageFolder(train_data_dir,transform = train_transforms)
        cubs200_test_dataset = Indices_ImageFolder(test_data_dir,transform = test_transforms)
        self.idx2class = {i:f'class {i}' for i in range(max(cubs200_train_dataset.targets)+1)}

        if isinstance(cubs200_train_dataset.targets, list):
            cubs200_train_dataset.targets = torch.Tensor(cubs200_train_dataset.targets).type(torch.int64) # Need to change into int64 to use in test step 
            cubs200_test_dataset.targets = torch.Tensor(cubs200_test_dataset.targets).type(torch.int64) # Need to change into int64 to use in test step 
        elif isinstance(cubs200_train_dataset.targets,np.ndarray):
            cubs200_train_dataset.targets = torch.from_numpy(cubs200_train_dataset.targets).type(torch.int64)  
            cubs200_test_dataset.targets = torch.from_numpy(cubs200_test_dataset.targets).type(torch.int64)  
        self.train_dataset, val_dataset = random_split(cubs200_train_dataset, [5500, 494],generator=torch.Generator().manual_seed(self.seed)
        )
        # splitting just to keep format consistent
        self.test_dataset, _ = random_split(cubs200_test_dataset, [5790, 4],generator=torch.Generator().manual_seed(self.seed)
        )

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
        cubs200_transforms = transform_lib.Compose([
            transforms.Resize(size = (32,32)),
            transform_lib.ToTensor(),
            cubs200_normalization()
        ])
        return cubs200_transforms

'''
datamodule = CUB200DataModule()
datamodule.setup()
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
import ipdb; ipdb.set_trace()
'''