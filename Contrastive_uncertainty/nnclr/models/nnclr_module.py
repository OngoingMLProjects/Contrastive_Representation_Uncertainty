from torch.utils import data
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch.distributed as dist
from typing import List, Tuple

from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.moco.models.resnet_models import custom_resnet18,custom_resnet34,custom_resnet50
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.general.utils.hybrid_utils import gather


class NNCLRModule(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        queue_size: int = 65536,
        softmax_temperature: float = 0.07,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
        instance_encoder:str = 'resnet50',
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

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
            nn.BatchNorm1d(self.hparams.emb_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
            nn.BatchNorm1d(self.hparams.emb_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
            nn.BatchNorm1d(self.hparams.emb_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
            nn.BatchNorm1d(self.hparams.emb_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim),
        )

        # queue
        self.register_buffer("queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("queue_y", -torch.ones(self.hparams.queue_size, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def name(self):
        ''' return name of model'''
        return model_names_dict['NNCLR']

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


    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor, y: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.
        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
        """

        z = gather(z)
        y = gather(y)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.hparams.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        self.queue_y[ptr : ptr + batch_size] = y  # type: ignore
        ptr = (ptr + batch_size) % self.hparams.queue_size

        self.queue_ptr[0] = ptr  # type: ignore
    
    @torch.no_grad()
    def find_nn(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbor of a sample.
        Args:
            z (torch.Tensor): a batch of projected features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """

        idx = (z @ self.queue.T).max(dim=1)[1]
        nn = self.queue[idx]
        return idx, nn

       
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
    
    def forward(self, im_1, im_2, targets):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        Output:
            logits, targets, proto_logits, proto_targets
        """        

        feats1, feats2 = self.encoder(im_1), self.encoder(im_2)
        z1, z2 = self.projector(feats1), self.projector(feats2)
        p1, p2 = self.predictor(z1), self.predictor(z2)

        z1, z2 = F.normalize(z1,dim=-1), F.normalize(z2,dim=-1)

        # find nn
        idx1, nn1 = self.find_nn(z1)
        _, nn2 = self.find_nn(z2)
        

        b = targets.size(0)
        nn_acc = (targets == self.queue_y[idx1]).sum() / b

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1, targets)

        
        return z1, z2, p1, p2, nn1, nn2, nn_acc
    
    def nnclr_loss_func(self, nn: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
        predicted features p from view 2.
        Args:
            nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
            p (torch.Tensor): NxD Tensor containing predicted features from view 2
            temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
                to 0.1.
        Returns:
            torch.Tensor: NNCLR loss.
        """

        nn = F.normalize(nn, dim=-1)
        p = F.normalize(p, dim=-1)

        logits = nn @ p.T / self.hparams.softmax_temperature

        n = p.size(0)
        labels = torch.arange(n, device=p.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def loss_function(self, batch):
        (img_1, img_2), *labels, indices = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        

        z1, z2, p1, p2, nn1, nn2, nn_acc = self(img_1, img_2,labels)
        loss = (self.nnclr_loss_func(nn1, p2) / 2
        + self.nnclr_loss_func(nn2, p1) / 2)

        metrics = {'Loss': loss, 'NN Accuracy':nn_acc}
                
        return metrics


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
    