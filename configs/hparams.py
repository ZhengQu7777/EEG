def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(dataset_name))
    return globals()[dataset_name]

class supervised():
    def __init__(self):
        super(supervised, self).__init__()
        self.train_params = {
            'num_epochs': 60,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'learning_rate': 1e-3,
            'codimnet_pinn_weight': 0.1,
            'lr_scheduler': 'step',
            'lr_step_size': 20,
            'lr_gamma': 0.5,
        }
