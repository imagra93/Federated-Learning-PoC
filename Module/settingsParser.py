import argparse

def args_parser_cen():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_classes', type=int, default=10)
    # Learning
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # Loss Functions
    parser.add_argument('--criterion', type=str, default='Nllloss')

    parser.add_argument('--iid', type=None, default=None)
    
    settings = parser.parse_args()
    return settings

def args_parser_fed():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_classes', type=int, default=10)
    # Learning
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--local_batch', type=int, default=8)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # Loss Functions
    parser.add_argument('--criterion', type=str, default='Nllloss')
    # Federated settings
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--unequal', type=int, default=0)
   
    settings = parser.parse_args()
    return settings

def args_parser_fed_3lvl():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_classes', type=int, default=10)
    # Learning
    parser.add_argument('--num_clusters', type=int, default=6)
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--local_batch', type=int, default=8)
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # Loss Functions
    parser.add_argument('--criterion', type=str, default='Nllloss')
    # Federated settings
    parser.add_argument('--global_rounds', type=int, default=3)
    parser.add_argument('--cluster_rounds', type=int, default=3)
    parser.add_argument('--num_clients', type=int, default=30)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--unequal', type=int, default=0)
   
    settings = parser.parse_args()
    return settings