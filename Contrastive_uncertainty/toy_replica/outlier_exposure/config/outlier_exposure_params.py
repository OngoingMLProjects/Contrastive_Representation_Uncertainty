from Contrastive_uncertainty.general.run.model_names import model_names_dict


outlier_exposure_hparams = dict(
emb_dim = 128,
instance_encoder = 'resnet50',

# optimizer args
optimizer = 'sgd',
learning_rate= 0.03,#3e-4,
momentum= 0.9,
weight_decay = 1e-4,

dataset = 'Blobs',
auxillary_dataset = ['TwoMoons'], #  Dataset for the auxillary dataset for the OE datamodule
OOD_dataset = ['TwoMoons'],


use_mlp = True,
pretrained_network = None,#'Pretrained_models/finetuned_network.pt',

model_type = model_names_dict['CE'],
project = 'toy_replica',  # evaluation, Moco_training
group = None,
notes = None,
)