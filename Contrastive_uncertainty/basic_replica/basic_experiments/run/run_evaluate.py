import wandb

# Import general params
from Contrastive_uncertainty.experiments.train.evaluate_experiments import evaluate
from Contrastive_uncertainty.experiments.config.trainer_params import trainer_hparams


# Code to obtain run paths from a project and group
run_paths = []
api = wandb.Api()

# Gets the runs corresponding to a specific filter
# https://github.com/wandb/client/blob/v0.10.31/wandb/apis/public.py
#https://github.com/wandb/client/blob/v0.12.1/wandb/apis/public.py#L752-L851

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines"}) # "OOD detection at different scales experiment" (other group I use to run experiments)

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.epochs":300})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.model_type":"Centroid_VicReg","config.dataset": "CIFAR10"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.model_type":"Centroid_VicReg","config.dataset": "CIFAR100"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.model_type":"Centroid_VicReg"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs":300})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs":300})
runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs":300,"$or": [{"config.dataset":"CIFAR100" }, {"config.dataset": "CIFAR10"}]})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "CE","config.epochs":300})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "CE","config.epochs":300,"$or": [{"config.dataset":"MNIST" }, {"config.dataset": "FashionMNIST"},{"config.dataset": "KMNIST"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs":300,"$or": [{"config.dataset":"MNIST" }, {"config.dataset": "FashionMNIST"},{"config.dataset": "KMNIST"}]})

# Used to filter simulations which are finished rather than still continuing
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing","config.model_type":"Centroid_VicReg",'state':'finished'})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"New Model Testing",'state':'finished'})



# Choose specifcally the specific group, the CIFAR100 dataset as well as choosing Moco or Supcon model
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.dataset": "CIFAR100","$or": [{"config.model_type":"Moco" }, {"config.model_type": "SupCon"}]})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "SupCon","config.epochs":300,"$or": [{"config.dataset":"CIFAR10" }, {"config.dataset": "CIFAR100"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"OOD hierarchy baselines","config.model_type": "Moco"})


#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco" }, {"config.model_type": "CE"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type": "SupCon"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats"})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"Moco"}, {"config.model_type": "SupCon"}]})
#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Separate branch combinations","config.branch_weights":[0,0,1]})

#runs = api.runs(path="nerdk312/evaluation", filters={"config.group":"Baselines Repeats","$or": [{"config.model_type":"CE"}, {"config.dataset": "MNIST"}, {"config.dataset": "FashionMNIST"}]})


for i in range(len(runs)):
    # Joins together the path of the runs which are separated into different parts in a list
    run_path = '/'.join(runs[i].path)
    run_paths.append(run_path)

evaluate(run_paths, trainer_hparams)
