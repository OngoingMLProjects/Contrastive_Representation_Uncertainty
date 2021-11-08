# Import general params
from Contrastive_uncertainty.experiments.config.batch_params import base1_hparams, trainer1_hparams
from Contrastive_uncertainty.experiments.train.automatic_train_experiments import train

train(base1_hparams,trainer1_hparams)

