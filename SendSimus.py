from types import SimpleNamespace
from main_fed import main

settings = SimpleNamespace()
# Dataset
settings.dataset = 'mnist'
settings.num_classes = 10
# Learning
settings.local_epoch = 10
settings.local_batch = 8
# Optimizer
settings.optimizer = 'sgd' # adam
settings.lr = 0.01
settings.momentum = 0.5
settings.weight_decay = 1e-4
# Loss Functions
settings.criterion = 'Nllloss'
# Federated settings
settings.rounds = 10
settings.num_clients = 10
settings.frac = 1
settings.iid = 3 # 'Default set to IID. Set to 0 for IID. 1 - low non-IID, 2 - medium non-IID, 3 - large non-IID'
settings.unequal = 0 # 'whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)'

# Call main.
main(settings)
