# Import general params
from Contrastive_uncertainty.experiments.config.practice_params import batch_practice_base_hparams, batch_practice_trainer_hparams
from Contrastive_uncertainty.experiments.train.train_experiments import train

train(batch_practice_base_hparams, batch_practice_trainer_hparams)
