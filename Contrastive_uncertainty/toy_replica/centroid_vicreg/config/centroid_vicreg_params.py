centroid_vicreg_hparams = dict(
emb_dim = 128,
num_negatives = 65536,
encoder_momentum = 0.999,
softmax_temperature = 0.07,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'Blobs',
OOD_dataset = ['TwoMoons','Diagonal'],

use_mlp = True,

loss_weights = [1,1,1],
callbacks = ['Model_saving'],

model_type = 'Centroid_VicReg',
project = 'toy_replica',# evaluation, Moco_training
group = None,
notes = None,
)