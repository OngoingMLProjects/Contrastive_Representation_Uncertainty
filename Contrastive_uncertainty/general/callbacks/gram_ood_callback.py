from re import I
#from tqdm import tqdm_notebook as tqdm

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

# Based on https://github.com/VectorInstitute/gram-ood-detection/blob/master/ResNet_Cifar10.ipynb
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

        self.classes = range(self.Datamodule.num_classes)

        self.OOD_dataname = self.OOD_Datamodule.name
        self.summary_key = f'Gram AUROC OOD {self.OOD_dataname}'
        self.summary_aupr = self.summary_key.replace("AUROC", "AUPR")
        self.summary_fpr = self.summary_key.replace("AUROC", "FPR") 
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.forward_callback(trainer=trainer, pl_module=pl_module) 

    def forward_callback(self,trainer,pl_module):
        self.OOD_Datamodule.setup() # SETUP AGAIN TO RESET AFTER PROVIDING THE TRANSFORM FOR THE DATA
        # train_logits, train_confs, train_preds,labels_train = self.get_predictions(pl_module,train_loader)
        # test_logits, test_confs, test_preds,labels_test = self.get_predictions(pl_module,test_loader)
        # ood_logits, ood_confs, ood_preds,labels_ood = self.get_predictions(pl_module,test_loader)

        self.Datamodule.batch_size = 1
        self.OOD_Datamodule.batch_size = 1
        
        updated_train_loader = self.Datamodule.deterministic_train_dataloader()
        
        updated_test_loader = self.Datamodule.test_dataloader()
        updated_ood_loader = self.OOD_Datamodule.test_dataloader()
        
        #updated_test_loader = self.Datamodule.deterministic_train_dataloader()
        #updated_ood_loader = self.OOD_Datamodule.deterministic_train_dataloader()
        
        data_train = list(updated_train_loader)
        data = list(updated_test_loader)
        ood_data = list(updated_ood_loader)
        self.compute_minmaxs(pl_module,data_train)
        #self.compare_deviations(pl_module,ood_data,ood_data)
        test_deviations = self.compute_test_deviations(pl_module,data)
        ood_deviations = self.compute_ood_deviations(pl_module,ood_data)
        self.get_eval_results(test_deviations, ood_deviations)
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
        train_preds = []
        train_confs = []
        train_logits = []
        for idx in range(0,len(data_train),128):
            batch = torch.squeeze(torch.stack([x[0][0] for x in data_train[idx:idx+128]]),dim=1).cuda()

            logits = pl_module.class_forward(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            logits = (logits.cpu().detach().numpy())

            train_confs.extend(np.max(confs,axis=1))    
            train_preds.extend(preds)
            train_logits.extend(logits)
        
        for PRED in self.classes:
            train_indices = np.where(np.array(train_preds)==PRED)[0]# obtain the predictions which were predicted a certain class
            if len(train_indices) > 0:
                train_PRED = torch.squeeze(torch.stack([data_train[i][0][0] for i in train_indices]),dim=1) # Get the data which had a certain prediction (num_samples, 1, height,width)
                #data_train[i] is the index, data_train[i][0] is the different augmented training data for the ith index and data_train[i][0][0] is the different versions for both cases
                mins,maxs = pl_module.encoder.get_min_max(train_PRED,power=POWERS) # list of values of size len(feat_l), where each entry has a size equal to the number of channels present
                self.mins[PRED] = self.cpu(mins)
                self.maxs[PRED] = self.cpu(maxs)
            else:
                self.mins[PRED] = []
                self.maxs[PRED] = []
            torch.cuda.empty_cache()

    def cpu(self,ob):
        for i in range(len(ob)):
            for j in range(len(ob[i])):
                ob[i][j] = ob[i][j].cpu()
        return ob

    def cuda(self,ob):
        for i in range(len(ob)):
            for j in range(len(ob[i])):
                ob[i][j] = ob[i][j].cuda()
        return ob

    def compute_test_deviations(self,pl_module,data,POWERS=[10]):
        test_preds = []
        test_confs = []
        test_logits = []

        for idx in range(0,len(data),128):
            batch = torch.squeeze(torch.stack([x[0][0] for x in data[idx:idx+128]]),dim=1).cuda()

            logits = pl_module.class_forward(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            logits = (logits.cpu().detach().numpy())

            test_confs.extend(np.max(confs,axis=1))    
            test_preds.extend(preds)
            test_logits.extend(logits)
        
        all_test_deviations = None
        test_classes = []
        #max_test_confs = np.argmax(test_confs,axis=1) # need to use the max confidence for the callback
        for PRED in self.classes:
            test_indices = np.where(np.array(test_preds)==PRED)[0]
            if len(test_indices) > 0:
                test_PRED = torch.squeeze(torch.stack([data[i][0][0] for i in test_indices]),dim=1) # shape (b, num_channels, height, width)
                test_confs_PRED = np.array([test_confs[i] for i in test_indices])
                #test_confs_PRED = np.array([max_test_confs[i] for i in test_indices]) # shape (batch, num classes)

                test_classes.extend([PRED]*len(test_indices))

                mins = self.cuda(self.mins[PRED])
                maxs = self.cuda(self.maxs[PRED])
                # get deviations has shape (batch, len(featL))
                test_deviations = pl_module.encoder.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis] # test_confs_PRED[:,np.newaxis] has shape (batch,1,num_classes)
                self.cpu(mins)
                self.cpu(maxs)
                if all_test_deviations is None:
                    all_test_deviations = test_deviations
                else:
                    all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
            
            torch.cuda.empty_cache()
        self.all_test_deviations = all_test_deviations
        return all_test_deviations
        #self.test_classes = np.array(test_classes)

    # Used for the purpose of debugging. Using different determinsitic dataloaders for the training data can 
    # give an AUROC which is different than OOD due to the data being different as a result of having a different random split.
    # When passing in the exact same data into compare deviations, I obtain an AUROC and AUPR of 0.5 which is the correct behaviour
    def compare_deviations(self,pl_module,data,ood,POWERS=[10]):
        test_preds = []
        test_confs = []
        test_logits = []

        for idx in range(0,len(data),128):
            batch = torch.squeeze(torch.stack([x[0][0] for x in data[idx:idx+128]]),dim=1).cuda()

            logits = pl_module.class_forward(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            logits = (logits.cpu().detach().numpy())

            test_confs.extend(np.max(confs,axis=1))    
            test_preds.extend(preds)
            test_logits.extend(logits)
        
        all_test_deviations = None
        test_classes = []
        #max_test_confs = np.argmax(test_confs,axis=1) # need to use the max confidence for the callback
        for PRED in self.classes:
            test_indices = np.where(np.array(test_preds)==PRED)[0]
            if len(test_indices) > 0:
                test_PRED = torch.squeeze(torch.stack([data[i][0][0] for i in test_indices]),dim=1) # shape (b, num_channels, height, width)
                test_confs_PRED = np.array([test_confs[i] for i in test_indices])
                #test_confs_PRED = np.array([max_test_confs[i] for i in test_indices]) # shape (batch, num classes)

                test_classes.extend([PRED]*len(test_indices))

                mins = self.cuda(self.mins[PRED])
                maxs = self.cuda(self.maxs[PRED])
                # get deviations has shape (batch, len(featL))
                #import ipdb; ipdb.set_trace()
                test_deviations = pl_module.encoder.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis] # test_confs_PRED[:,np.newaxis] has shape (batch,1,num_classes)
                self.cpu(mins)
                self.cpu(maxs)
                if all_test_deviations is None:
                    all_test_deviations = test_deviations
                else:
                    all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
            
            torch.cuda.empty_cache()
        self.all_test_deviations = all_test_deviations

        ood_preds = []
        ood_confs = []
        
        for idx in range(0,len(ood),128):
            batch = torch.squeeze(torch.stack([x[0][0] for x in ood[idx:idx+128]]),dim=1).cuda()
            logits = pl_module.class_forward(batch) # shape (batch,num_classes)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy() # shape (batch, num_classes) 
            preds = np.argmax(confs,axis=1) # shape (batch,)
            
            ood_confs.extend(np.max(confs,axis=1)) # list of shape (datasize) where it adds a batch each time to the list
            ood_preds.extend(preds) # list of values of size (datasize) where it adds a batch each time to the list
            torch.cuda.empty_cache()
        print("Done")
        ood_classes = []
        all_ood_deviations = None
        for PRED in self.classes:
            ood_indices = np.where(np.array(ood_preds)==PRED)[0]
            if len(ood_indices)==0:
                continue
            ood_classes.extend([PRED]*len(ood_indices))
            
            ood_PRED = torch.squeeze(torch.stack([ood[i][0][0] for i in ood_indices]),dim=1) # shape (data size, num_channels, height, width)
            ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices]) # shape (datasize, )
            mins = self.cuda(self.mins[PRED])
            maxs = self.cuda(self.maxs[PRED])
            
            ood_deviations = pl_module.encoder.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]  # ood_confs_PRED[:,np.newaxis] changes the shape from (datasize,) into (datasize,1) which allows broadcasting to occur
            self.cpu(self.mins[PRED])
            self.cpu(self.maxs[PRED])            
            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
            torch.cuda.empty_cache()
            
        self.ood_classes = np.array(ood_classes)


        dtest = all_test_deviations.sum(axis=1)
        dood = all_ood_deviations.sum(axis=1)

        auroc, aupr, fpr = get_measures(dood,dtest)
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr
        # the test set and the OOD data have a shape (datasize, 40)
        # The data is obtained by summing along the axis to get a shape (datasize,)
        
    
    def compute_ood_deviations(self,pl_module,ood,POWERS=[10]):
        ood_preds = []
        ood_confs = []
        
        for idx in range(0,len(ood),128):
            batch = torch.squeeze(torch.stack([x[0][0] for x in ood[idx:idx+128]]),dim=1).cuda()
            logits = pl_module.class_forward(batch) # shape (batch,num_classes)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy() # shape (batch, num_classes) 
            preds = np.argmax(confs,axis=1) # shape (batch,)
            
            ood_confs.extend(np.max(confs,axis=1)) # list of shape (datasize) where it adds a batch each time to the list
            ood_preds.extend(preds) # list of values of size (datasize) where it adds a batch each time to the list
            torch.cuda.empty_cache()
        print("Done")
        ood_classes = []
        all_ood_deviations = None
        for PRED in self.classes:
            ood_indices = np.where(np.array(ood_preds)==PRED)[0]
            if len(ood_indices)==0:
                continue
            ood_classes.extend([PRED]*len(ood_indices))
            
            ood_PRED = torch.squeeze(torch.stack([ood[i][0][0] for i in ood_indices]),dim=1) # shape (data size, num_channels, height, width)
            ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices]) # shape (datasize, )
            mins = self.cuda(self.mins[PRED])
            maxs = self.cuda(self.maxs[PRED])
            
            ood_deviations = pl_module.encoder.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)/ood_confs_PRED[:,np.newaxis]  # ood_confs_PRED[:,np.newaxis] changes the shape from (datasize,) into (datasize,1) which allows broadcasting to occur
            self.cpu(self.mins[PRED])
            self.cpu(self.maxs[PRED])            
            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
            torch.cuda.empty_cache()
            
        self.ood_classes = np.array(ood_classes)
        # the test set and the OOD data have a shape (datasize, 40)
        # The data is obtained by summing along the axis to get a shape (datasize,)
        return ood_deviations
        #self.detect(self.all_test_deviations, all_ood_deviations)
        
        
    def get_eval_results(self, test_deviations, ood_deviations):
        
        dtest = test_deviations.sum(axis=1)
        dood = ood_deviations.sum(axis=1)
        
        auroc, aupr, fpr = get_measures(dood,dtest)
        wandb.run.summary[self.summary_key] = auroc
        wandb.run.summary[self.summary_aupr] = aupr
        wandb.run.summary[self.summary_fpr] = fpr
        
    
    
    '''
    def detect(self,all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
        average_results = {}
        for i in range(1,11):
            random.seed(i)

            validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
            test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))

            validation = all_test_deviations[validation_indices]
            test_deviations = all_test_deviations[test_indices] 
            import ipdb; ipdb.set_trace()

            t95 = validation.mean(axis=0)+10**-7
            if not normalize:
                t95 = np.ones_like(t95)
            test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
            ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
            import ipdb; ipdb.set_trace()
    '''