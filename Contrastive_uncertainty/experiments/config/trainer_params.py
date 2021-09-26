from pytorch_lightning import callbacks

trainer_hparams = dict(
# Miscellaneous arguments
seed = 26,
epochs = 0,
bsz = 64,
# Trainer configurations
fast_run = False,
quick_callback = False,
training_ratio = 1.0,
validation_ratio = 1.0,
test_ratio = 1.0,
val_check = 1,  # evaluation, Moco_training
callbacks = ['Oracle Nearest 10 Neighbours Class 1D Typicality'],
#callbacks = ['Nearest','Class Mahalanobis','Mahalanobis Distance'],
#callbacks = ['Nearest Neighbours Class 1D Typicality'],

#callbacks = ['Nearest Class Neighbours'],
#callbacks = ['Nearest Neighbours 1D Typicality'],
#callbacks = ['Class Mahalanobis','Mahalanobis OOD Fractions', 'Nearest Neighbours'],
#callbacks = ['Model_saving']
)
