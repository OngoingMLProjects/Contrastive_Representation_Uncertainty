from numpy.core.numeric import indices
from scipy.special.orthogonal import orthopoly1d
import torch
from torch.autograd import grad
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import sklearn.metrics as skm
import faiss
import statistics 
import scipy

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.callbacks.gradcam.pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from Contrastive_uncertainty.general.callbacks.gradcam.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# Based on code from https://github.com/jacobgil/pytorch-grad-cam
class Cam_Visualization(pl.Callback):
    def __init__(self, Datamodule,
        quick_callback:bool = True):

        super().__init__()
        self.Datamodule = Datamodule

        #self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly
        
    def on_test_epoch_end(self, trainer, pl_module):
        torch.set_grad_enabled(True) # need to set grad enabled true during the test step
        pl_module.eval()
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self,trainer,pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        
        self.get_visualization(pl_module, train_loader)

    def get_visualization(self, pl_module, dataloader):
        loader = quickloading(self.quick_callback, dataloader)
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            

            img = img.to(pl_module.device)

            target_layers = [pl_module.encoder.layer4[-1]]

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = None

            cam = GradCAM(pl_module.encoder,target_layers=target_layers,use_cuda= True)
            cam.batch_size = 16

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=img)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0,:]
            #visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            

            ''' Need to change this part        
            rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]) # change to tensor and normalize the image
            '''
