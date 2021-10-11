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

from Contrastive_uncertainty.moco.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean, min_distance_accuracy
from Contrastive_uncertainty.general.run.model_names import model_names_dict

class CentroidVICRegModule(pl.LightningModule):
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
        instance_encoder:str = 'resnet50',
        pretrained_network:str = None,
        invariance_weight:float = 1.0,
        variance_weight:float = 1.0,
        covariance_weight:float = 1.0
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        self.save_hyperparameters()
        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.num_channels = datamodule.num_channels
        
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = self.init_encoders()
        
        self.projector = nn.Sequential(
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
            nn.BatchNorm1d(self.hparams.emb_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
            nn.BatchNorm1d(self.hparams.emb_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
        )
        # obtain the centroids of the data
        self.mean_vectors= self.load_centroids()
    
    def load_centroids(self):
        if self.num_classes == 10:
            kernel_dict = loadmat('meanvar1_featuredim128_class10.mat') # Nawid - load precomputed centres
        else: 
            kernel_dict = loadmat('meanvar1_featuredim128_class100.mat') # Nawid -Used for the case of CIFAR100 training dataset
        mean_vectors = kernel_dict['mean_logits'] #num_class X num_dense # Nawid - centres
        mean_vectors = torch.from_numpy(mean_vectors).type(torch.FloatTensor).detach() # Detach so no gradient is backpropagated for this tensor and make into float32
        mean_vectors = mean_vectors.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        #normalized_mean_vectors = nn.functional.normalize(mean_vectors, dim=1) #  Mean vectors are already normalized
        return mean_vectors

    @property
    def name(self):
        ''' return name of model'''
        return model_names_dict['Centroid_VicReg']

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        if self.hparams.instance_encoder == 'resnet18':
            print('using resnet18')
            encoder = custom_resnet18(latent_size = self.hparams.emb_dim,num_channels = self.num_channels,num_classes = self.num_classes)
            
        elif self.hparams.instance_encoder =='resnet50':
            print('using resnet50')
            encoder = custom_resnet50(latent_size = self.hparams.emb_dim,num_channels = self.num_channels,num_classes = self.num_classes)
        
        return encoder

       
    def callback_vector(self, x): # vector for the representation before using separate branches for the task
        """
        Input:
            x: a batch of images for classification
        Output:
            z: latent vector
        """
        z = self.encoder(x)
        z = nn.functional.normalize(z, dim=1)
        return z
    

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
        loss_variance = self.variance_loss(z1, z2)
        loss_covariance = self.covariance_loss(z1,z2)

        preds = self.class_predictions(z1)
        return loss_invariance, loss_variance, loss_covariance, preds

    def loss_function(self, batch):
        (img_1, img_2), *labels, indices = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        
        loss_invariance, loss_variance, loss_covariance, preds = self(img_1, img_2,labels)
        acc = min_distance_accuracy(preds,labels)
        #loss = (loss_invariance* self.hparams.invariance_weight) + (loss_variance* self.hparams.variance_weight) + (loss_covariance* self.hparams.covariance_weight)
        # Training with the same weight for that specific hyperparameter
        loss = (loss_invariance* self.hparams.invariance_weight) + (loss_variance* self.hparams.invariance_weight) + (loss_covariance* self.hparams.covariance_weight)

        #acc_1, acc_5 = precision_at_k(output, target,top_k=(1,5))
        metrics = {'Loss': loss, 'Invariance Loss': loss_invariance, 'Variance Loss':loss_variance, 'Covariance Loss':loss_covariance, 'Accuracy':acc}
        return metrics
    
    def invariance_loss(self, z1: torch.Tensor,z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:e
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """
        
        
        centroids = self.mean_vectors[labels]
        return 0.5*(F.mse_loss(z1, centroids) + F.mse_loss(z2, centroids)) 

    
    def class_predictions(self,z):
        diff = [z - self.mean_vectors[class_num] for class_num in range(self.num_classes)]
        distances = [(diff[class_num]**2).mean(1) for class_num in range(self.num_classes)]
        y_pred = torch.stack(distances,dim=1)
        
        return y_pred


    def variance_loss(self,z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """
        

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        return std_loss
    
    def covariance_loss(self,z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """

        N, D = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
        return cov_loss


    def training_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Loss']
        return loss
        
    def validation_step(self, batch, batch_idx,dataset_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        

    def configure_optimizers(self):
        if self.hparams.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer
    
