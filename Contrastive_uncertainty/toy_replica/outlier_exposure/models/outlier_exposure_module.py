import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from Contrastive_uncertainty.toy_replica.cross_entropy.models.encoder_model import Backbone
from Contrastive_uncertainty.general.utils.hybrid_utils import label_smoothing, LabelSmoothingCrossEntropy 
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.general.run.model_names import model_names_dict


class OutlierExposureToy(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        oe_datamodule: pl.LightningDataModule = None,
        pretrained_network:str = None,
        ):

        super().__init__()
        # Nawid - required to use for the fine tuning
        #self.save_hyperparameters()

        self.oe_datamodule = oe_datamodule
        self.num_classes = self.oe_datamodule.num_classes
        self.num_channels = self.oe_datamodule.num_channels
        
        self.emb_dim = emb_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = self.init_encoders()
        

    @property
    def name(self):
        ''' return name of model'''
        return model_names_dict['OE']

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(20, self.emb_dim)
        encoder.class_fc2 = nn.Linear(self.emb_dim, self.num_classes)
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
    
    def class_forward(self, x):
        z = self.encoder(x)
        z = nn.functional.normalize(z, dim=1)
        logits = self.encoder.class_fc2(z)
        return logits

    def loss_function(self, batch):
        #import ipdb; ipdb.set_trace()
        (img_1, img_2), *labels, indices = batch
        
        # Takes into account if it has coarse labels
        # Using * makes it into a list (so the length of the list is related to how many different labels types there are)
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        '''
        if len(labels) > 1:
            labels = labels[0]
        '''
        
        logits = self.class_forward(img_1)
        
        loss = F.cross_entropy(logits.float(), labels.long())

        class_acc1, class_acc5 = precision_at_k(logits, labels, top_k=(1, 5))
        metrics = {'Class Loss': loss, 'Class Accuracy @ 1': class_acc1, 'Class Accuracy @ 5': class_acc5}

        return metrics


    def training_step(self, batch, batch_idx):
        import ipdb; ipdb.set_trace()
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Training ' + k, v.item(),on_epoch=True)
        loss = metrics['Class Loss']
        return loss
        

    def validation_step(self, batch, batch_idx,dataset_idx):
        import ipdb; ipdb.set_trace()
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Validation ' + k, v.item(),on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self.loss_function(batch)
        for k,v in metrics.items():
            if v is not None: self.log('Test ' + k, v.item(),on_epoch=True)
        
        
        #return {'logits':logits,'target':labels} # returns y_pred as y_pred are essentially the logits in this case, and I want to log how the logits change in time
     
    def configure_optimizers(self):
        if self.optimizer =='sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.learning_rate,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.optimizer =='adam':
            optimizer = torch.optim.Adam(self.parameters(), self.learning_rate,
                                        weight_decay=self.weight_decay)
        return optimizer
    