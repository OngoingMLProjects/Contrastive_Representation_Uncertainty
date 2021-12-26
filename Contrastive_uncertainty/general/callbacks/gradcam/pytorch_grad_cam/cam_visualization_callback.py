import torch
from torch.autograd import grad
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import sklearn.metrics as skm
from PIL import Image

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
        #train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader() # use test loader as there is less data augmentation for the test loader
        
        self.get_visualization(trainer,pl_module, test_loader)

    def get_visualization(self,trainer, pl_module, dataloader):
        loader = quickloading(self.quick_callback, dataloader)
        

        
        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            
            img = img.to(pl_module.device)
            #target_layers = [pl_module.encoder.layer4[-1]]
            target_layers = [pl_module.encoder.layer4[-1]]

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = None

            cam = GradCAM(pl_module.encoder,target_layers=target_layers,use_cuda= True)
            cam.batch_size = 32

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=img)
            

            # In this example grayscale_cam has only one image in the batch:
            #grayscale_cam = grayscale_cam[0,:]
            rgb_img = img.data.cpu().numpy()
            rgb_img = rgb_img.reshape(rgb_img.shape[0],rgb_img.shape[2],rgb_img.shape[3],rgb_img.shape[1])
            mean, std = self.Datamodule.test_transforms.normalization.mean, self.Datamodule.test_transforms.normalization.std 
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam,mean, std, use_rgb=True)            
            # https://docs.wandb.ai/guides/track/log/media
            images = [Image.fromarray(image) for image in visualization]
          
            wandb.log({"GradCam Heatmaps": [wandb.Image(image) for image in images]})
            
            # visualization = torch.tensor(visualization) # shape (batch, H,W,C)
            # 
            # visualization = visualization.reshape(visualization.shape[0],visualization.shape[3], visualization.shape[1], visualization.shape[2]) # shape (batch,C, H,W)
            # visualization = visualization.to(dtype=torch.float32)
            # grid_imgs = torchvision.utils.make_grid(visualization)    
            # images = wandb.Image(grid_imgs, caption="Top: Input, Bottom: Counterfactual")
            # wandb.log({self.logging: images})
            
            # '''
            # https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image            
            # img = Image.fromarray(visualization[0], 'RGB')
            # img.show()
            # '''