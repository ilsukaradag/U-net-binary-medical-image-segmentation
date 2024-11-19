import torch
import numpy as np
import argparse

def save_model(model, save_path):
    '''
    Saves the state dictionary of a PyTorch model to a specified path.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to save.
    - save_path (str): The path where the model's state dictionary will be saved.
    '''
    torch.save(model.state_dict(), save_path)

def set_seed(seed):
    '''
    Sets the random seed for reproducibility in NumPy and PyTorch.
    
    Args:
    - seed (int): The seed value for random number generators.
    
    Notes:
    - Ensures that the results are reproducible by fixing the seed for various random number generators.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_arg_parser():
    '''
    Argument parser for training configuration.
    
    Returns:
    - args: Parsed arguments, including device, experiment ID, learning rate, batch size, number of epochs, and mode.
    '''
    parser = argparse.ArgumentParser()
    
    # Device configuration (e.g., 'cuda:0' or 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Experiment ID for saving checkpoints and results
    parser.add_argument('--exp_id', type=str, default='exp/0')
    
    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=3e-3)  # Corrected default from string to float
    
    # Batch size for training
    parser.add_argument('--bs', type=int, default=10)
    
    # Number of epochs for training
    parser.add_argument('--epoch', type=int, default=20)
    
    # Mode for running the script (e.g., 'train' or 'test')
    parser.add_argument('--mode', type=str, default='train')
    
    args = parser.parse_args()
    return args

def test_arg_parser():
    '''
    Argument parser for testing configuration.
    
    Returns:
    - args: Parsed arguments, including device, model path, and mode.
    '''
    parser = argparse.ArgumentParser()
    
    # Device configuration (e.g., 'cuda:0' or 'cpu')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Path to the saved model for testing
    parser.add_argument('--model_path', type=str, default='')
    
    # Experiment ID to use latest saved model 
    parser.add_argument('--exp_id', type=str, default='exp/0')
    
    parser.add_argument('--bs', type=int, default=8, help='Batch size for testing') 
    
    args = parser.parse_args()
    return args
