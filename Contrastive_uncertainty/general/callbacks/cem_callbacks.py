from math import log
from numpy.core.numeric import indices
from numpy.lib.financial import _ipmt_dispatcher
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torchvision
import wandb
import sklearn.metrics as skm
import faiss
import statistics 


import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score


from Contrastive_uncertainty.general.utils.hybrid_utils import OOD_conf_matrix
from Contrastive_uncertainty.general.callbacks.general_callbacks import quickloading
from Contrastive_uncertainty.general.utils.pl_metrics import precision_at_k, mean
from Contrastive_uncertainty.general.run.model_names import model_names_dict
from Contrastive_uncertainty.general.callbacks.ood_callbacks import get_roc_sklearn


class ContrastiveExplanationMethod(pl.Callback):
    def __init__(self, Datamodule,
        kappa: float = 10.0,
        c_init: float = 10.0,
        c_converge: float = 0.1,
        beta: float = 0.1,
        iterations: int = 1000,
        n_searches: int = 3, #9,
        learning_rate: float = 0.01,
        input_shape: tuple = (1, 28, 28),
        quick_callback:bool = True):
        
        """
        Initialise the CEM model.
        
        mode
            for pertinant negatives 'PN' or for pertinant positives 'PP'.
        
        kappa
            confidence parameter used in the loss functions (eq. 2)
            and (eq. 4) in the original paper.
        const
            initial regularisation coefficient for the attack loss term.
        beta
            regularisation coefficent for the L1 term of the optimisation
            objective.
        iterations
            number of iterations in each search
        n_searches
            number of searches, also the number of times c gets adjusted
        learning_rate
            initial learning rate used to optimise the slack variable
        input_shape
            shape of single input sample, used to reshape for classifier
            and ae input
        """

        super().__init__()
        self.Datamodule = Datamodule
        self.quick_callback = quick_callback # Quick callback used to make dataloaders only use a single batch of the data in order to make the testing process occur quickly    

        self.kappa = kappa
        self.c_converge = c_converge
        self.c_init = c_init
        self.beta = beta

        self.iterations = iterations
        self.n_searches = n_searches
        self.learning_rate = learning_rate
    
        self.input_shape = input_shape
    
    def on_test_epoch_end(self, trainer, pl_module):
        # Perform callback only for the situation
        torch.set_grad_enabled(True) # need to set grad enabled true during the test step
        pl_module.eval()
        self.explain_callback(trainer,pl_module)
    
    # Performs all the computation in the callback
    def explain_callback(self,trainer,pl_module):
        test_loader = self.Datamodule.test_dataloader()
        self.get_explanation(trainer,pl_module,test_loader)

    # Scores when the inputs are not perturbed by a gradient    
    def get_explanation(self,trainer,pl_module,dataloader):
        collated_delta = []
        loader = quickloading(self.quick_callback, dataloader)
        
        for index, (imgs, *label, indices) in enumerate(loader):

            assert len(loader)>0, 'loader is empty'
            if isinstance(imgs, tuple) or isinstance(imgs, list):
                    imgs, *aug_imgs = imgs # Used to take into accoutn whether the data is a tuple of the different augmentations
            
            imgs = imgs.to(pl_module.device)
            # Only get counterfactuals for the first batch
            if index == 0:
                for img in imgs: # Go through individual imgs
                    delta = self.explain(img,pl_module)
                    collated_delta.append(delta)

                #import ipdb; ipdb.set_trace()
                collated_delta = torch.cat(collated_delta)
                collated_imgs = torch.cat((imgs,collated_delta))
                grid_imgs = torchvision.utils.make_grid(collated_imgs,nrow=self.Datamodule.batch_size)    
                images = wandb.Image(grid_imgs, caption="Top: Input, Bottom: Counterfactual")
                wandb.log({"examples": images})
            else:
                break

                #img = img.to(pl_module.device)
                

            ''' Enables plotting the imgs and the counterfactuals on top of one another  
            #delta = delta[0]
            deltas = torch.cat(8*[delta])
            img = img[0:8].to(pl_module.device)
            imgs = torch.cat((img,deltas))
            #imgs = torch.stack((img,delta))
            import ipdb; ipdb.set_trace()
            grid_imgs = torchvision.utils.make_grid(imgs)
            images = wandb.Image(grid_imgs, caption="Top: Input, Bottom: Counterfactual")
            wandb.log({"examples": images})
            '''
        return delta
    
    # Input images should be zero-mean and only needs to be rescaled when using an autoencoder which I am not using 
    def explain(self, orig, pl_module, mode="PN"):
        """
        Determine pertinents for a given input sample.

        orig
            The original input sample to find the pertinent for.
        mode
            Either "PP" for pertinent positives or "PN" for pertinent
            negatives.

        """
        if mode not in ["PN", "PP"]:
            raise ValueError(
                "Invalid mode. Please select either 'PP' or 'PN' as mode.")
        
        const = self.c_init
        step = 0
        # May also need to change the shape of the modle
        orig = orig.unsqueeze(0) # Nawid - unsqueeze a single sample to make it into the correct shape
        #orig = orig.to(pl_module.device)


        # Nawid - set initial loss as infinity and have a best delta
        best_loss = float("inf")
        best_delta = None
        
        # Nawid - original classification prediction
        orig_output = pl_module.class_forward(orig)

        # mask for the originally selected label (t_0)
        target_mask = torch.zeros(orig_output.shape).to(pl_module.device)
        target_mask[torch.arange(orig_output.shape[0]),
                    torch.argmax(orig_output)] = 1 # Nawid - place values of 1 at the locations of the true labels

        # mask for the originally non-selected labels (i =/= t_0)
        nontarget_mask = torch.ones(orig_output.shape).to(
            pl_module.device) - target_mask # Nawid - place 1s at the values where the original target is not used 


        # Nawid - iterate for a number of searches
        for search in range(self.n_searches):

            found_solution = False

            adv_img = torch.zeros(orig.shape).to(pl_module.device) # Nawid - shape of adversarial image
            adv_img_slack = torch.zeros(orig.shape).to(
                pl_module.device).detach().requires_grad_(True) 

            # optimise for the slack variable y, with a square root decaying
            # learning rate
            optim = torch.optim.SGD([adv_img_slack], lr=self.learning_rate)
            # Nawid - number of iterations
            for step in range(1, self.iterations + 1):
                print('step:',step)
                # - Optimisation objective; (eq. 1) and (eq. 3) - #

                # reset the computational graph
                optim.zero_grad()
                adv_img_slack.requires_grad_(True)

                # Optimise for image + delta, this is more stable (negative sample)
                delta = orig - adv_img # Nawid - difference between the original image and the perturbed image (delta)
                delta_slack = orig - adv_img_slack #Nawid - difference between original image and slack variable

                if mode == "PP":
                    perturbation_score = pl_module.class_forward(
                        delta_slack.view(-1, *self.input_shape))
                elif mode == "PN":
                    perturbation_score = pl_module.class_forward(
                        adv_img_slack.view(-1, *self.input_shape))

                target_lab_score = torch.max(
                    target_mask * perturbation_score) # Nawid - score for the actual class it belongs to
                nontarget_lab_score = torch.max(
                    nontarget_mask * perturbation_score) # Nawd - score for the different possible counterfactual classes

                # classification objective loss (eq. 2)
                if mode == "PP": # Nawid - objective for the pertinent positives and the pertinent negatives
                    loss_attack = const * torch.max(
                        torch.tensor(0.).to(pl_module.device),
                        nontarget_lab_score - target_lab_score + self.kappa
                        ) 
                elif mode == "PN":
                    loss_attack = const * torch.max(
                        torch.tensor(0.).to(pl_module.device),
                        -nontarget_lab_score + target_lab_score + self.kappa
                        ) # Nawid - This is a max between 0 and  target_lab_score - nontarget_lab_score + self.kappa which is the same as a max between target_lab_score - nontarget_lab_score and -kappa 
                
                # if the attack loss has converged to 0, a viable solution
                # has been found!
                if loss_attack.item() == 0:
                    found_solution = True

                # L2 regularisation term (eq. 1)
                l2_loss = torch.sum(delta ** 2) # Nawid - L2 regularisation

                # reconstruction loss (eq. 1). reshape the image to fit ae
                # input, reshape the output of the autoencoder back. Since our
                # images are zero-mean, scale back to original MNIST range

                # final optimisation objective
                loss_to_optimise = loss_attack + l2_loss 

                # optimise for the slack variable, adjust lr
                loss_to_optimise.backward()
                optim.step()
                # Nawid - slow down the learing rate 
                optim.param_groups[0]['lr'] = (self.learning_rate - 0.0) *\
                    (1 - step/self.iterations) ** 0.5
                # Nawid - Optimizer seems to be updated frequently

                adv_img_slack.requires_grad_(False)

                # - FISTA and corresponding update steps (eq. 5, 6) - #
                with torch.no_grad():

                    # Shrinkage thresholding function (eq. 7)
                    # Nawid -checks that adv_img_slack - orig is greater than beta
                    cond1 = torch.gt(
                        adv_img_slack - orig, self.beta
                        ).type(torch.float)
                    # Nawid -checks that adv_img_slack - orig is less than or equal to beta
                    cond2 = torch.le(
                        torch.abs(adv_img_slack - orig), self.beta
                        ).type(torch.float)
                    # Nawid -checks that adv_img_slack - orig is less than negative beta
                    cond3 = torch.lt(
                        adv_img_slack - orig, -self.beta
                        ).type(torch.float)

                    # Nawid - Ensure that the delta values are not too large
                    # Ensure all delta values are between -0.5 and 0.5
                    upper = torch.min(adv_img_slack - self.beta,
                                      torch.tensor(0.5).to(pl_module.device))
                    lower = torch.max(adv_img_slack + self.beta,
                                      torch.tensor(-0.5).to(pl_module.device))

                    # Nawid - assigns values for delta(k+1) 
                    assign_adv_img = (
                        cond1 * upper + cond2 * orig + cond3 * lower) # Nawid - performs element wise shirinkage based on shrinkage

                    # Apply projection to the slack variable to obtain
                    # the value for delta (eq. 5)
                    # Nawid - Use different conditions whether I am geerating positives or negatves
                    cond4 = torch.gt( # Nawid- checks whether it is less than or more than the original
                        assign_adv_img - orig, 0).type(torch.float) 
                    cond5 = torch.le(
                        assign_adv_img - orig, 0).type(torch.float)
                    if mode == "PP":
                        assign_adv_img = cond5 * assign_adv_img + cond4 * orig
                    elif mode == "PN":
                        assign_adv_img = cond4 * assign_adv_img + cond5 * orig # Nawid - obtain the values which are more than the original and remove values less than original

                    # Apply momentum from previous delta and projection step
                    # to obtain the value for the slack variable (eq. 6)
                    mom = (step / (step + 3)) * (assign_adv_img - adv_img) #Nawid - second part of equation six where assign_adv_img is delta(k+1) and adv_img is delta(k)
                    assign_adv_img_slack = assign_adv_img + mom # Nawid  - equaion 6
                    cond6 = torch.gt(
                        assign_adv_img_slack - orig, 0).type(torch.float)
                    cond7 = torch.le(
                        assign_adv_img_slack - orig, 0).type(torch.float)

                    # For PP only retain delta values that are smaller than
                    # the corresponding feature values in the original image
                    if mode == "PP": # Nawid - updating according to a momentum update
                        assign_adv_img_slack = (
                            cond7 * assign_adv_img_slack +
                            cond6 * orig)
                    # For PN only retain delta values that are larger than
                    # the corresponding feature values in the original image
                    elif mode == "PN": # Nawid - https://blogs.princeton.edu/imabandit/2013/04/11/orf523-ista-and-fista/
                        assign_adv_img_slack = (
                            cond6 * assign_adv_img_slack +
                            cond7 * orig) # corresponds to second last equation on blog post, updating according to the momentum term

                    adv_img.data.copy_(assign_adv_img) # Nawid - update the adversarial image (delta k+1)
                    adv_img_slack.data.copy_(assign_adv_img_slack) # Nawid - update the slack variable  (yk+1)

                    # check if the found delta solves the classification
                    # problem, retain if it is the most regularised solution
                    if loss_attack.item() == 0: # Nawid - checks if the loss is zero which leads to updating the delta with adv_img
                        if loss_to_optimise < best_loss:
                            best_loss = loss_to_optimise
                            best_delta = adv_img.detach().clone()

                            print("new best delta found, loss: {}".format(
                                    loss_to_optimise))
            # If in this search a solution has been found we can decrease the
            # weight of the attack loss to increase the regularisation, else
            # increase c to decrease regularisation
            if found_solution:
                const = (self.c_converge + const) / 2
            else:
                const *= 10


        return best_delta




class ContrastiveExplanationDistance(ContrastiveExplanationMethod):
    def __init__(self, Datamodule,
        kappa: float = 0.0,#10.0,
        c_init: float = 0.0,#10.0,
        c_converge: float = 0.1,
        beta: float = 0.1,
        iterations: int = 10,#1000,
        n_searches: int = 2, #9,
        learning_rate: float = 0.01,
        input_shape: tuple = (1, 28, 28),
        quick_callback:bool = True):

        super().__init__(Datamodule,
        kappa,
        c_init,
        c_converge,
        beta,
        iterations,
        n_searches,
        learning_rate,
        input_shape,
        quick_callback)

    def on_test_epoch_end(self, trainer, pl_module):
        return super().on_test_epoch_end(trainer, pl_module)

    def explain_callback(self, trainer, pl_module):
        return super().explain_callback(trainer, pl_module)
    
    def explain_callback(self,trainer,pl_module):
        train_loader = self.Datamodule.deterministic_train_dataloader()
        test_loader = self.Datamodule.test_dataloader()
        self.get_centroids(pl_module, train_loader)
        self.get_explanation(trainer,pl_module,test_loader)
    
    def get_centroids(self,pl_module,dataloader):
        features, labels = [], []
        with torch.no_grad():
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
                feature_vector = pl_module.callback_vector(img) # Performs the callback for the desired level

                features.append(feature_vector)
                labels.append(label)

                #features.append(feature_vector)
                #labels.append(label)

        collated_features = torch.cat(features)
        collated_labels = torch.cat(labels)
        xc = [collated_features[collated_labels == i] for i in np.unique(collated_labels)]
        
        # Add all the data to represent the background class
        xc.append(collated_features)

        centroids = [torch.mean(x,dim=0) for x in xc]
        # get the centroids for each class including the background class
        # Centroids for the different classes including the background class
        self.centroids = torch.stack(centroids) # shape (num_classes +1, embedding dim)

        self.centroids = nn.functional.normalize(self.centroids, dim=1)
    
    def explain(self, orig, pl_module, mode='PN'):
        """
        Determine pertinents for a given input sample.

        orig
            The original input sample to find the pertinent for.
        mode
            Either "PP" for pertinent positives or "PN" for pertinent
            negatives.

        """
        if mode not in ["PN", "PP"]:
            raise ValueError(
                "Invalid mode. Please select either 'PP' or 'PN' as mode.")

        const = self.c_init
        step = 0
        # May also need to change the shape of the modle
        orig = orig.unsqueeze(0) # Nawid - unsqueeze a single sample to make it into the correct shape
        #orig = orig.to(pl_module.device)

        # Nawid - set initial loss as infinity and have a best delta
        best_loss = float("inf")
        best_delta = None

        # obtain the output vector of the model
        embedding_vector = pl_module.callback_vector(orig)
        #output = pl_module.class_forward(orig)
        orig_output = torch.mm(embedding_vector,self.centroids.t()) #logits for each of the classes
        target_mask = torch.zeros(orig_output .shape).to(pl_module.device)
        target_mask[torch.arange(orig_output .shape[0]),-1] = 1 # Nawid - place values of 1 at the location of the last index which represent OOD data
        
        nontarget_mask = torch.ones(orig_output.shape).to(
            pl_module.device) - target_mask # Nawid - place 1s at the values where the original target is not used 
        

        '''
        - Need to obtain prototypes of the data
	    - Need to calculate a loss based on a prototype (which should give me a target mask
	    - Make a mask for the correct class which wll have a 1
	    - Make a mask for the OOD sample which should have a 1
    
	    OR
	    - Make a mask which gives the OOD class a value of 1 (OOD data point)
	    - Make a mask which gives the all the other classes a value of 1 

	    - Get the scores for the perturbed input (for the OOD class)
	    - Get the score for the perturbed input for the ID class and obtain the max
	    - Optimize the objective    
        '''

        # Nawid - iterate for a number of searches
        for search in range(self.n_searches):

            found_solution = False

            adv_img = torch.zeros(orig.shape).to(pl_module.device) # Nawid - shape of adversarial image
            adv_img_slack = torch.zeros(orig.shape).to(
                pl_module.device).detach().requires_grad_(True) 

            # optimise for the slack variable y, with a square root decaying
            # learning rate
            optim = torch.optim.SGD([adv_img_slack], lr=self.learning_rate)
            # Nawid - number of iterations
            for step in range(1, self.iterations + 1):
                print('step:',step)
                # - Optimisation objective; (eq. 1) and (eq. 3) - #

                # reset the computational graph
                optim.zero_grad()
                adv_img_slack.requires_grad_(True)

                # Optimise for image + delta, this is more stable (negative sample)
                delta = orig - adv_img # Nawid - difference between the original image and the perturbed image (delta)
                delta_slack = orig - adv_img_slack #Nawid - difference between original image and slack variable

                if mode == "PP":
                    embedding = pl_module.callback_vector(delta_slack.view(-1, *self.input_shape))
                    perturbation_score = torch.mm(embedding,self.centroids.t()) #logits for each of the classes

                    #perturbation_score = pl_module.class_forward(
                        #delta_slack.view(-1, *self.input_shape))
                elif mode == "PN":
                    embedding = pl_module.callback_vector(adv_img_slack.view(-1, *self.input_shape))
                    perturbation_score = torch.mm(embedding,self.centroids.t()) #logits for each of the classes
                    #perturbation_score = pl_module.class_forward(
                    #    adv_img_slack.view(-1, *self.input_shape))

                target_lab_score = torch.max(
                    target_mask * perturbation_score) # Nawid - score for the actual class it belongs to
                nontarget_lab_score = torch.max(
                    nontarget_mask * perturbation_score) # Nawd - score for the different possible counterfactual classes

                # classification objective loss (eq. 2)
                if mode == "PP": # Nawid - objective for the pertinent positives and the pertinent negatives
                    loss_attack = const * torch.max(
                        torch.tensor(0.).to(pl_module.device),
                        nontarget_lab_score - target_lab_score + self.kappa
                        ) 
                elif mode == "PN":
                    loss_attack = const * torch.max(
                        torch.tensor(0.).to(pl_module.device),
                        -nontarget_lab_score + target_lab_score + self.kappa
                        ) # Nawid - This is a max between 0 and  target_lab_score - nontarget_lab_score + self.kappa which is the same as a max between target_lab_score - nontarget_lab_score and -kappa 
                
                # if the attack loss has converged to 0, a viable solution
                # has been found!
                if loss_attack.item() == 0:
                    found_solution = True

                # L2 regularisation term (eq. 1)
                l2_loss = torch.sum(delta ** 2) # Nawid - L2 regularisation

                # reconstruction loss (eq. 1). reshape the image to fit ae
                # input, reshape the output of the autoencoder back. Since our
                # images are zero-mean, scale back to original MNIST range

                # final optimisation objective
                loss_to_optimise = loss_attack + l2_loss 

                # optimise for the slack variable, adjust lr
                loss_to_optimise.backward()
                optim.step()
                # Nawid - slow down the learing rate 
                optim.param_groups[0]['lr'] = (self.learning_rate - 0.0) *\
                    (1 - step/self.iterations) ** 0.5
                # Nawid - Optimizer seems to be updated frequently

                adv_img_slack.requires_grad_(False)

                # - FISTA and corresponding update steps (eq. 5, 6) - #
                with torch.no_grad():

                    # Shrinkage thresholding function (eq. 7)
                    # Nawid -checks that adv_img_slack - orig is greater than beta
                    cond1 = torch.gt(
                        adv_img_slack - orig, self.beta
                        ).type(torch.float)
                    # Nawid -checks that adv_img_slack - orig is less than or equal to beta
                    cond2 = torch.le(
                        torch.abs(adv_img_slack - orig), self.beta
                        ).type(torch.float)
                    # Nawid -checks that adv_img_slack - orig is less than negative beta
                    cond3 = torch.lt(
                        adv_img_slack - orig, -self.beta
                        ).type(torch.float)

                    # Nawid - Ensure that the delta values are not too large
                    # Ensure all delta values are between -0.5 and 0.5
                    upper = torch.min(adv_img_slack - self.beta,
                                      torch.tensor(0.5).to(pl_module.device))
                    lower = torch.max(adv_img_slack + self.beta,
                                      torch.tensor(-0.5).to(pl_module.device))

                    # Nawid - assigns values for delta(k+1) 
                    assign_adv_img = (
                        cond1 * upper + cond2 * orig + cond3 * lower) # Nawid - performs element wise shirinkage based on shrinkage

                    # Apply projection to the slack variable to obtain
                    # the value for delta (eq. 5)
                    # Nawid - Use different conditions whether I am geerating positives or negatves
                    cond4 = torch.gt( # Nawid- checks whether it is less than or more than the original
                        assign_adv_img - orig, 0).type(torch.float) 
                    cond5 = torch.le(
                        assign_adv_img - orig, 0).type(torch.float)
                    if mode == "PP":
                        assign_adv_img = cond5 * assign_adv_img + cond4 * orig
                    elif mode == "PN":
                        assign_adv_img = cond4 * assign_adv_img + cond5 * orig # Nawid - obtain the values which are more than the original and remove values less than original

                    # Apply momentum from previous delta and projection step
                    # to obtain the value for the slack variable (eq. 6)
                    mom = (step / (step + 3)) * (assign_adv_img - adv_img) #Nawid - second part of equation six where assign_adv_img is delta(k+1) and adv_img is delta(k)
                    assign_adv_img_slack = assign_adv_img + mom # Nawid  - equaion 6
                    cond6 = torch.gt(
                        assign_adv_img_slack - orig, 0).type(torch.float)
                    cond7 = torch.le(
                        assign_adv_img_slack - orig, 0).type(torch.float)

                    # For PP only retain delta values that are smaller than
                    # the corresponding feature values in the original image
                    if mode == "PP": # Nawid - updating according to a momentum update
                        assign_adv_img_slack = (
                            cond7 * assign_adv_img_slack +
                            cond6 * orig)
                    # For PN only retain delta values that are larger than
                    # the corresponding feature values in the original image
                    elif mode == "PN": # Nawid - https://blogs.princeton.edu/imabandit/2013/04/11/orf523-ista-and-fista/
                        assign_adv_img_slack = (
                            cond6 * assign_adv_img_slack +
                            cond7 * orig) # corresponds to second last equation on blog post, updating according to the momentum term

                    adv_img.data.copy_(assign_adv_img) # Nawid - update the adversarial image (delta k+1)
                    adv_img_slack.data.copy_(assign_adv_img_slack) # Nawid - update the slack variable  (yk+1)

                    # check if the found delta solves the classification
                    # problem, retain if it is the most regularised solution
                    if loss_attack.item() == 0: # Nawid - checks if the loss is zero which leads to updating the delta with adv_img
                        if loss_to_optimise < best_loss:
                            best_loss = loss_to_optimise
                            best_delta = adv_img.detach().clone()

                            print("new best delta found, loss: {}".format(
                                    loss_to_optimise))
            # If in this search a solution has been found we can decrease the
            # weight of the attack loss to increase the regularisation, else
            # increase c to decrease regularisation
            if found_solution:
                const = (self.c_converge + const) / 2
            else:
                const *= 10


        return best_delta