import math
import numpy as np
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
from Contrastive_uncertainty.general.run.model_names import model_names_dict


# Based on https://github.com/WangFeng18/InvariancePropagation
# Used to upweigh the different terms in the loss function
def GaussianRampUp(i_epoch, end_epoch, weight=5):
	m = max(1-i_epoch/end_epoch, 0)**2
	v = np.exp(-weight * m)
	return v

def BinaryRampUp(i_epoch, end_epoch):
	return int(i_epoch > end_epoch)

def l2_normalize(x):
	return x / torch.sqrt(torch.sum(x**2, dim=1).unsqueeze(1))

class MemoryBank_v1(object):
    def __init__(self, n_points, emb_dim,k, device, m=0.5):
        super().__init__()
        self.m = m
        self.emb_dim = emb_dim
        self.device = device
        self.n_points = n_points
        self.points = torch.zeros(n_points, ).to(device).detach()
        self.cluster_number = 0
        self.point_centroid = None
        self.k = k
        self.neigh = torch.zeros(n_points, self.k, dtype=torch.long).to(device).detach()
        self.neigh_sim = torch.zeros(n_points, self.k).to(device).detach()
    
    def clear(self):
        self.points = torch.zeros(self.n_points, self.emb_dim).to(self.device).detach()
    
    def random_init_bank(self):
        stdv = 1. / math.sqrt(128/3)
        self.points = torch.rand(self.n_points, self.emb_dim).mul_(2*stdv).add_(-stdv).to(self.device).detach()
    
    def update_points(self, points, point_indices):
        norm_points = l2_normalize(points)
        data_replace = self.m * self.points[point_indices,:] + (1-self.m) * norm_points
        self.points[point_indices,:] = l2_normalize(data_replace)
    
    def get_all_dot_products(self, points):
        assert len(points.size()) == 2
        return torch.matmul(points, torch.transpose(self.points, 1, 0))
'''
TODO:
Make it so that I can access the current epoch to enable ramping up the invariance loss term
Change the name of the function using model name dict
Make it so that all the nearest neighbours belong to the same class

'''
class InvariancePropagationToy(pl.LightningModule):
    def __init__(self,
        emb_dim: int = 128,
        n_background: int = 4096,
        k:int = 4, # number of nearest neighbours for propagation
        diffusion_layer:int = 3,
        hard_pos:bool = True,
        exclusive:bool = True,       
        softmax_temperature: float = 0.07,
        ramp_up_type = 'binary',
        lam_inv: float = 0.6, # Used to control the ramp up
        optimizer:str = 'sgd',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        datamodule: pl.LightningDataModule = None,
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory_bank = MemoryBank_v1(self.datamodule.num_samples,self.hparams.emb_dim,self.hparams.k, device,m=0.5)

        # Initialise the memory bank
        self.memory_bank.random_init_bank()
        
        if self.hparams.ramp_up_type =='binary':
            self.ramp_up = lambda i_epoch: BinaryRampUp(i_epoch, 30)
        elif self.hparams.ramp_up_type == 'gaussian':
            self.ramp_up = lambda i_epoch: GaussianRampUp(i_epoch, 30, 5)
        elif self.hparams.ramp_up_type == 'zero':
            self.ramp_up = lambda i_epoch: 1

    @property
    def name(self):
        ''' return name of model'''
        return 'InvariancePropagation' #model_names_dict['Centroid_VicReg']

    def init_encoders(self):
        """
        Override to add your own encoders
        """
        encoder = Backbone(20, self.hparams.emb_dim)
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

    def update_nn(self, background_indices, point_indices): # update the nearest neighbours
        nei = background_indices[:, :self.hparams.k+1] # obtain the nearest neighbours
        condition = (nei == point_indices.unsqueeze(dim=1))
        backup = nei[:, self.hparams.k:self.hparams.k+1].expand_as(nei)
        nei_exclusive = torch.where(
			condition,
			backup,
			nei,
		)
        nei_exclusive = nei_exclusive[:, :self.hparams.k]
        self.memory_bank.neigh[point_indices] = nei_exclusive # update the nearest neighbours
    
    def propagate(self, point_indices): # obtain the different nearest neighbours
        cur_point = 0
        matrix = self.memory_bank.neigh[point_indices] # 256 x 4
        end_point = matrix.size(1) - 1
        layer = 2
        while layer <= self.hparams.diffusion_layer: # I believe diffusion layer is the number of values to obtain
            current_nodes = matrix[:, cur_point] # 256
            sub_matrix = self.memory_bank.neigh[current_nodes] # 256 x 4
            matrix = torch.cat([matrix, sub_matrix], dim=1)
            
            if cur_point == end_point:
                layer += 1
                end_point = matrix.size(1) - 1
            cur_point += 1
        return matrix
    
    def _exp(self, dot_prods): # calculate exp
        # Z = 2876934.2 / 1281167 * self.data_len
        return torch.exp(dot_prods / self.hparams.softmax_temperature) # t is the temperature term I believe

    def forward(self, img, point_indices, return_neighbour=False):
        points = self.encoder(img)
        norm_points = l2_normalize(points) # normalize the point
        all_sim = self._exp(self.memory_bank.get_all_dot_products(norm_points)) # calculate coine similarity of all the data points
        self_sim = all_sim[list(range(all_sim.size(0))), point_indices] # calculate the similarity of the different dataa points
        background_sim, background_indices = all_sim.topk(k=self.hparams.n_background, dim=1, largest=True, sorted=True)
        instance_loss = -(self_sim/background_sim.sum(dim=1) + 1e-7).log().mean() # calculate the instance discrimination loss I believe
        
        # invariance propagation
        neighs = self.propagate(point_indices, self.memory_bank) # calculate the neighbours
        
        InvPropLoss = 0
        background_exclusive_sim = background_sim.sum(dim=1) - self_sim # find the similarity with the nearest neighbours, remove the self similarity terms
        ## one
        pos_sim = torch.gather(all_sim, index=neighs, dim=1)
        if self.hparams.hard_pos:
            hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=False, sorted=True) # obtain the smallest similarity
        else:
            hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=True, sorted=True) # obtain the largest similarity
        
        if self.hparams.exclusive:	
            InvPropLoss = -( hard_pos_sim.sum(dim=1) / background_exclusive_sim + 1e-7).log().mean() # Does not include self similarity
        else:
            InvPropLoss = -( hard_pos_sim.sum(dim=1) / (background_exclusive_sim + self_sim) + 1e-7).log().mean()
        
        self.update_nn(background_indices, point_indices, self.memory_bank)
        if return_neighbour:
            return instance_loss, InvPropLoss, neighs, torch.gather(neighs, index=hp_indices, dim=1)
        else:
            return instance_loss, InvPropLoss


    def loss_function(self, batch):
        (img_1, img_2), *labels, indices = batch
        if isinstance(labels, tuple) or isinstance(labels, list):
            labels, *coarse_labels = labels
        
        L_ins, L_inv = self(img_1, indices)
        
        loss = L_ins + (self.hparams.lam_inv * self.ramp_up(i_epoch) * L_inv) #  obtain the epoch number 

        metrics = {'Loss': loss, 'Instance Loss': L_ins, 'Invariance Loss':L_inv}
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

'''
class InvariancePropagationLoss(nn.Module):
	def __init__(self, t, n_background=4096, diffusion_layer=3, k=4, n_pos=50, exclusive=True, InvP=True, hard_pos=True):
		super(InvariancePropagationLoss, self).__init__()
		self.t = t
		self.n_background = n_background
		self.diffusion_layer = diffusion_layer
		self.k = k
		self.n_pos = n_pos
		self.exclusive = exclusive
		self.InvP = InvP
		self.hard_pos = hard_pos
		
		print('DIFFUSION_LAYERS: {}'.format(self.diffusion_layer))
		print('K_nearst: {}'.format(self.k))
		print('N_POS: {}'.format(self.n_pos))

	

	def propagate(self, point_indices, memory_bank): # obtain the different nearest neighbours
		cur_point = 0
		matrix = memory_bank.neigh[point_indices] # 256 x 4
		end_point = matrix.size(1) - 1
		layer = 2
		while layer <= self.diffusion_layer: # I believe diffusion layer is the number of values to obtain
			current_nodes = matrix[:, cur_point] # 256

			sub_matrix = memory_bank.neigh[current_nodes] # 256 x 4
			matrix = torch.cat([matrix, sub_matrix], dim=1)

			if cur_point == end_point:
				layer += 1
				end_point = matrix.size(1) - 1
			cur_point += 1
		return matrix


	def forward(self, points, point_indices, memory_bank, return_neighbour=False):
		norm_points = l2_normalize(points) # normalize the point
		all_sim = self._exp(memory_bank.get_all_dot_products(norm_points)) # calculate coine similarity of all the data points
		self_sim = all_sim[list(range(all_sim.size(0))), point_indices] # calculate the similarity of the different dataa points
		background_sim, background_indices = all_sim.topk(k=self.n_background, dim=1, largest=True, sorted=True)

		lossA = -(self_sim/background_sim.sum(dim=1) + 1e-7).log().mean() # calculate the instance discrimination loss I believe

		if self.InvP:
			# invariance propagation
			neighs = self.propagate(point_indices, memory_bank) # calculate the neighbours

			lossB = 0
			background_exclusive_sim = background_sim.sum(dim=1) - self_sim # find the similarity with the nearest neighbours, remove the self similarity terms

			## one
			pos_sim = torch.gather(all_sim, index=neighs, dim=1)
			if self.hard_pos:
				hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=False, sorted=True) # obtain the smallest similarity
			else:
				hard_pos_sim, hp_indices = pos_sim.topk(k=min(self.n_pos, pos_sim.size(1)), dim=1, largest=True, sorted=True) # obtain the largest similarity

			if self.exclusive:	
				lossB = -( hard_pos_sim.sum(dim=1) / background_exclusive_sim + 1e-7).log().mean() # Does not include self similarity
			else:
				# print('no exclusive')
				lossB = -( hard_pos_sim.sum(dim=1) / (background_exclusive_sim + self_sim) + 1e-7).log().mean()

		else:
			lossB = 0.0
			neighs = None

		self.update_nn(background_indices, point_indices, memory_bank)

		if return_neighbour:
			return lossA, lossB, neighs, torch.gather(neighs, index=hp_indices, dim=1)
		else:
			return lossA, lossB

	def _exp(self, dot_prods): # calculate exp
		# Z = 2876934.2 / 1281167 * self.data_len
		return torch.exp(dot_prods / self.t) # t is the temperature term I believe
'''