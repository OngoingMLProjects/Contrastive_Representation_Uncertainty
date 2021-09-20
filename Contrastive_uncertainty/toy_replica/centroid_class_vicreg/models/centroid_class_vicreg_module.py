from torch.utils import data
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from scipy.io import loadmat
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from Contrastive_uncertainty.toy_replica.moco.models.encoder_model import Backbone
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean, min_distance_accuracy
from Contrastive_uncertainty.toy_replica.centroid_vicreg.models.centroid_vicreg_module import CentroidVICRegToy



class CentroidClassVICRegToy(CentroidVICRegToy):
    def __init__(self,
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        invariance_weight: float = 1.0,
        variance_weight: float = 1.0,
        covariance_weight: float = 1.0
        ):

        super().__init__(emb_dim,num_negatives, encoder_momentum,
        softmax_temperature, optimizer, learning_rate, momentum,
        weight_decay, datamodule, 
        invariance_weight,variance_weight, covariance_weight)

    @property
    def name(self):
        ''' return name of model'''
        return 'CentroidClassVICReg'
    

    # Pass through the img through 
    def forward(self, img1, img2, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets, proto_logits, proto_targets
        """        
        # compute query features
        z1 = self.encoder(img1)  # queries: NxC
        z1 = self.projector(z1)
        z1 = nn.functional.normalize(z1, dim=1) # Nawid - normalised query embeddings


        z2 = self.encoder(img2)  # queries: NxC
        z2 = self.projector(z2)
        z2 = nn.functional.normalize(z2, dim=1) # Nawid - normalised query embeddings

        loss_invariance = self.invariance_loss(z1, z2, labels)
        loss_variance = self.variance_loss(z1, z2,labels)
        loss_covariance = self.covariance_loss(z1,z2)

        preds = self.class_predictions(z1)
        return loss_invariance, loss_variance, loss_covariance, preds


    def variance_loss(self,z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            labels (torch.Tensor): N Tensor containing labels for the different classes.
        Returns:
            torch.Tensor: variance regularization loss.
        """
        
        # Obtain the unique class values
        #unique_vals = torch.unique(labels,sorted=True)
        unique_vals = torch.unique(labels)

        class_z1 = [z1[labels ==i] for i in unique_vals]
        class_z2 = [z2[labels ==i] for i in unique_vals]

        eps = 1e-4
        # calculate the variance for all the data points in the same class
        class_std_z1 = [torch.sqrt(class_z1[class_num].var(dim=0) + eps) for class_num in range(len(class_z1))]
        class_std_z2 = [torch.sqrt(class_z2[class_num].var(dim=0) + eps) for class_num in range(len(class_z2))]
        class_std_z1, class_std_z2 = torch.stack(class_std_z1), torch.stack(class_std_z2) # stack the tensor to make it shape (num_class_present, embdim)

        # Calculate mean along the different class dimensions
        std_z1 = torch.mean(class_std_z1,dim=0)
        std_z2 = torch.mean(class_std_z2,dim=0)

        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        return std_loss