# Function which instantiates cross entropy model
def ModelInstance(model_module,config,oe_datamodule):
    model = model_module(emb_dim = config['emb_dim'], 
            optimizer = config['optimizer'],learning_rate = config['learning_rate'],
            momentum = config['momentum'], weight_decay = config['weight_decay'],
            oe_datamodule = oe_datamodule,
            instance_encoder = config['instance_encoder'])

    return model