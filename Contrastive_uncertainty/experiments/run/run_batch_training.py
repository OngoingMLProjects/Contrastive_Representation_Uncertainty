# Import general params
import sys
from Contrastive_uncertainty.experiments.config.batch_params import batch_base_hparams, batch_trainer_hparams, batch_base_hparams_1, batch_trainer_hparams_1, batch_base_hparams_2, batch_trainer_hparams_2
from Contrastive_uncertainty.experiments.train.train_experiments import train


# Uses the train function
def run_batch(argv):
    value = int(sys.argv[1])
    print('Value chosen:',value)
    if value == 0:
        train(batch_base_hparams, batch_trainer_hparams)
    elif value == 1:
        train(batch_base_hparams_1, batch_trainer_hparams_1)
    elif value == 2:
        train(batch_base_hparams_2, batch_trainer_hparams_2)
if __name__ =="__main__":
    run_batch(sys.argv)
    