from tqdm import tqdm_notebook as tqdm

import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import sklearn.metrics as skm

from Contrastive_uncertainty.general.utils.ood_utils import get_measures # Used to calculate the AUROC, FPR and AUPR
from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
#from Contrastive_uncertainty.Contrastive.models.loss_functions import class_discrimination
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean

class Gram_OOD(pl.Callback):
    def __init__(self,Datamodule,OOD_Datamodule,
        quick_callback:bool = True):
        super().__init__()

        self.Datamodule = Datamodule
        self.OOD_Datamodule = OOD_Datamodule
        self.OOD_Datamodule.test_transforms = self.Datamodule.test_transforms #  Make the transform of the OOD data the same as the actual data
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly

        self.all_test_deviations = None
        self.mins = {}
        self.maxs = {}

        self.classes = self.Datamodule.num_classes 
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA

        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        ood_loader = self.OOD_Datamodule.test_dataloader()

        train_logits, train_confs, train_preds,labels_train = self.get_predictions(pl_module,train_loader)
        test_logits, test_confs, test_preds,labels_test = self.get_predictions(pl_module,test_loader)
        ood_logits, ood_confs, ood_preds,labels_ood = self.get_predictions(pl_module,test_loader)

        # Obtain the predictions
        # Change code which goes from cpu for gpu into normal version

    def get_predictions(self,pl_module,dataloader):
        collated_logits,collated_preds, collated_confs, labels = [],[],[],[]

        loader = quickloading(self.quick_callback, dataloader)

        for index, (img, *label, indices) in enumerate(loader):
            assert len(loader)>0, 'loader is empty'
            if isinstance(img, tuple) or isinstance(img, list):
                    img, *aug_img = img # Used to take into accoutn whether the data is a tuple of the different augmentations
            
            # Selects the correct label based on the desired label level
            if len(label) > 1:
                label_index = 0
                label = label[label_index]
            else: # Used for the case of the OOD data
                label = label[0]

            img = img.to(pl_module.device)
            
            # Compute feature vector and place in list
            
            logits = pl_module.class_forward(img) # Performs the callback for the desired level
            #confidence = F.softmax(logits,dim=1)

            confidence = F.softmax(logits,dim=1).data.cpu().detach().numpy()
            predictions = np.argmax(confidence,axis=1)
            
            collated_logits += list(logits.data.cpu().detach().numpy())
            collated_confs += list(confidence)
            collated_preds += list(predictions)
            labels += list(label.data.cpu().numpy())
        
        return np.array(collated_logits), np.array(collated_confs), np.array(collated_preds), np.array(labels)
    
    def compute_minmaxs(self,pl_module,data_train,POWERS=[10]):
        for PRED in tqdm(self.classes):
            train_indices = np.where(np.array(train_preds)==PRED)[0] # obtain the predictions which were predicted a certain class
            train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]),dim=1) # Get the data which had a certain prediction
            mins,maxs = pl_module.get_min_max(train_PRED,power=POWERS)
            self.mins[PRED] = cpu(mins)
            self.maxs[PRED] = cpu(maxs)
            torch.cuda.empty_cache()

    def compute_test_deviations(self,pl_module,POWERS=[10]):
        all_test_deviations = None
        test_classes = []
        for PRED in tqdm(self.classes):
            test_indices = np.where(np.array(test_preds)==PRED)[0]
            test_PRED = torch.squeeze(torch.stack([data[i][0] for i in test_indices]),dim=1)
            test_confs_PRED = np.array([test_confs[i] for i in test_indices])
            
            test_classes.extend([PRED]*len(test_indices))
            
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            test_deviations = pl_module.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis]
            cpu(mins)
            cpu(maxs)
            if all_test_deviations is None:
                all_test_deviations = test_deviations
            else:
                all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
            torch.cuda.empty_cache()
        self.all_test_deviations = all_test_deviations
        
        #self.test_classes = np.array(test_classes)

    
    def compute_ood_deviations(self,pl_module,ood,POWERS=[10]):
        ood_preds = []
        ood_confs = []
        
        for idx in range(0,len(ood),128):
            batch = torch.squeeze(torch.stack([x[0] for x in ood[idx:idx+128]]),dim=1).cuda()
            logits = pl_module(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            
            ood_confs.extend(np.max(confs,axis=1))
            ood_preds.extend(preds)  
            torch.cuda.empty_cache()
        print("Done")
        
        ood_classes = []
        all_ood_deviations = None
        for PRED in tqdm(self.classes):
            ood_indices = np.where(np.array(ood_preds)==PRED)[0]
            if len(ood_indices)==0:
                continue
            ood_classes.extend([PRED]*len(ood_indices))
            
            ood_PRED = torch.squeeze(torch.stack([ood[i][0] for i in ood_indices]),dim=1)
            ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices])
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            ood_deviations = torch_model.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]
            cpu(self.mins[PRED])
            cpu(self.maxs[PRED])            
            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
            torch.cuda.empty_cache()
            
        self.ood_classes = np.array(ood_classes)
        
        average_results = detect(self.all_test_deviations,all_ood_deviations)
        return average_results, self.all_test_deviations, all_ood_deviations
    '''
    