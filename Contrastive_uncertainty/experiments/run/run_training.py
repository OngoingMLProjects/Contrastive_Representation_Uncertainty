# Import general params
from Contrastive_uncertainty.experiments.config.base_params import batch_base_hparams, batch_trainer_hparams
from Contrastive_uncertainty.experiments.train.train_experiments import train

train(batch_base_hparams,batch_trainer_hparams)
