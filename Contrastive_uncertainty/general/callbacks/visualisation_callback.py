import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import sklearn.metrics as skm
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from scipy.spatial.distance import cdist # Required for TwoMoons visualisation involving pairwise distances

class Visualisation(pl.Callback): # General class for visualisation
    def __init__(self, datamodule, 
        quick_callback:bool = True):

        self.datamodule = datamodule
        #self.ood_datamodule = ood_datamodule
        #self.ood_datamodule.test_transforms= self.datamodule.test_transforms   # Make it so that the OOD datamodule has the same transform as the true module

        
        #self.ood_datamodule.setup()
        # setup data
        self.quick_callback = quick_callback
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.datamodule.setup()
        num_classes = self.datamodule.num_classes

        # Obtain representations for the normal case as well as the concatenated representations
        representations, labels = self.obtain_representations(pl_module)
        # PCA visalisation
        self.pca_visualisation(representations, labels, f'inliers:',num_classes)
        # T-SNE visualisation
        self.tsne_visualisation(representations, labels, f'inliers',num_classes)

        # +1 in num classes to represent outlier data, *2 represents class specific outliers
        #self.pca_visualisation(concat_representations, concat_labels,config['num_classes']+1,'general')

        # Obtain the concatenated representations for the case where the different values can be used for the task
    
    def obtain_representations(self, pl_module):  # separate from init so that two moons does not make representations automatically using dataloader rather it uses X_vis
        # self.data = self.datamodule.test_dataloader() # instantiate val dataloader

        self.data = self.datamodule.test_dataset
        dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=200, shuffle=False, num_workers=6, pin_memory=False
            ) # Dataloader with batch size 500 to ensure that all the data is obtained for the tas
        
        #self.labels = self.datamodule.test_dataset.targets  # Obtain the test dataset targets

        loader = quickloading(self.quick_callback, dataloader)
        self.representations, self.labels = self.compute_representations(pl_module, loader)
        return self.representations, self.labels
    
    @torch.no_grad()
    def compute_representations(self, pl_module, loader):
        features, collated_labels = [],[]

        for i, (images, *labels, indices) in enumerate(tqdm(loader)): # Obtain data and labels from dataloader
            assert len(loader) >0, 'loader is empty'
            if isinstance(images, tuple) or isinstance(images, list):
                images, *aug_imgs = images
            
            # Selects the correct label based on the desired label level
            if len(labels) > 1:
                label_index = 0
                labels = labels[label_index]
            else: # Used for the case of the OOD data
                labels = labels[0] # Obtain feature vector

            images = images.to(pl_module.device)  # cuda(non_blocking=True)
            feature_vector = pl_module.callback_vector(images) # Performs the callback for the desired level

            features.append(feature_vector.cpu())
            collated_labels.append(labels.cpu())

        features = torch.cat(features)
        collated_labels = torch.cat(collated_labels)

        # Centroid calculation
        # Calculate a list for the centroids, which is then concatenated together
        centroids = [torch.mean(features[collated_labels==i],dim=0,keepdim=True) for i in torch.unique(collated_labels)] 
        centroids = torch.cat(centroids)
        # Concatenate the features and the centroids, as well as adding additional features for the centroid
        features = torch.cat((features, centroids))
        collated_labels = torch.cat((collated_labels,torch.unique(collated_labels))) 
        
        return features, collated_labels

    def pca_visualisation(self, representations, labels, name,num_classes):
        # Font size for annotation
        font_size = 15 if num_classes <100 else 10
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(representations)
        pca_one = pca_result[:,0]
        pca_two = pca_result[:,1]
        pca_three = pca_result[:,2]
        # 2D plot
        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x=pca_one, y=pca_two,
        hue = labels,
        palette=sns.color_palette('hls',n_colors=num_classes),
        legend="full",
        alpha=0.3
        )
        
        # Values of +1 required to make it move backwards effectively
        for class_num in range(num_classes):
            plt.annotate(f'{num_classes - (class_num+1)}', xy= (pca_one[-(class_num+1)],pca_two[-(class_num+1)]),fontsize=font_size)
        
        #plt.show()
        
        # Limits for te plot

        #sns.plt.xlim(-2.5,2.5)
        #sns.plt.ylim(-2.5,2.5)

        pca_filename = 'Images/'+name + '_data_pca.png'
        plt.savefig(pca_filename)
        wandb_pca = name +' PCA of Features'
        wandb.log({wandb_pca:wandb.Image(pca_filename)})
        plt.close()
        # 3D PCA plot
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
        xs=pca_one,
        ys=pca_two,
        zs=pca_three,
        c=labels,
        cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        for class_num in range(num_classes):
            ax.text(pca_one[-(class_num+1)], pca_two[-(class_num+1)], pca_three[-(class_num+1)], f"{num_classes - (class_num+1)}")
        '''
        for class_num in range(num_classes):
            plt.annotate(f'{num_classes - (class_num+1)}', xy= (pca_one[-(class_num+1)],pca_two[-(class_num+1)]),fontsize=20)
        '''
        #limits for plot to ensure fixed scale
        #ax.xlim(-2.5,2.5)
        #ax.ylim(-2.5,2.5)
        pca_3D_filename = 'Images/'+name + ' data_pca_3D.png'
        plt.savefig(pca_3D_filename)
        wandb_3D_pca = name + ' 3D PCA of Features'
        wandb.log({wandb_3D_pca: wandb.Image(pca_3D_filename)})
        plt.close()

    def tsne_visualisation(self, representations, labels, name,num_classes):
        font_size = 15 if num_classes <100 else 10
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(representations)
        tsne_one = tsne_results[:,0]
        tsne_two = tsne_results[:,1]

        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x=tsne_one, y=tsne_two,
        hue=labels,
        palette=sns.color_palette("hls", num_classes),
        legend="full",
        alpha=0.3
        )
        # https://matplotlib.org/stable/gallery/mplot3d/text3d.html
        for class_num in range(num_classes):
            plt.annotate(f'{num_classes - (class_num+1)}', xy= (tsne_one[-(class_num+1)],tsne_two[-(class_num+1)]),fontsize=font_size)
        
        # Used to control the scale of a seaborn plot
        #sns.plt.ylim(-15, 15)
        #sns.plt.xlim(-15, 15)

        tsne_filename = 'Images/'+ name +'_data_tsne.png'
        plt.savefig(tsne_filename)
        wandb_tsne = name +' TSNE of Features'
        wandb.log({wandb_tsne:wandb.Image(tsne_filename)})
        plt.close()

