import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''
    
    no_of_samples_to_show = min(4, len(images)) 
    fig, axes = plt.subplots(no_of_samples_to_show, 3, figsize=(12, no_of_samples_to_show* 4))

    outputs =  torch.sigmoid(outputs) > 0.5

    for index in range(no_of_samples_to_show):
        image = images[index].squeeze().cpu().numpy()
        mask = masks[index].squeeze().cpu().numpy()
        pred_mask = outputs[index].squeeze().cpu().numpy()

        axes[index, 0].imshow(image, cmap='gray')
        axes[index, 0].set_title('Original Image')
        axes[index, 0].axis('off')

        axes[index, 1].imshow(mask, cmap='gray')
        axes[index, 1].set_title('Ground Truth Mask')
        axes[index, 1].axis('off')

        axes[index, 2].imshow(pred_mask, cmap='gray')
        axes[index, 2].set_title('Predicted Mask')
        axes[index, 2].axis('off')

    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, f'epoch_{epoch}_batch_{batch_idx}.jpg')
    plt.tight_layout()
    plt.savefig(plot_filename, format='jpg')
    plt.close()
    print(f"Visualization saved to {plot_filename}")


def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''
    plt.figure(figsize=(10, 5))
    # Plot training loss
    plt.plot(train_loss_history, label='Training Loss', color='blue', marker='o')
    
    # Plot validation loss
    plt.plot(val_loss_history, label='Validation Loss', color='red', marker='o')
    
    # Label for x-axis
    plt.xlabel('Epochs')
    
    # Label for y-axis
    plt.ylabel('Loss')
    
    # Title for the plot
    plt.title('Training vs Validation Loss')
    
    # Display legend
    plt.legend()

    plt.grid(True)
    
    os.makedirs(plot_dir, exist_ok=True)
    # Save the figure to the 'figures' folder

    file_name = f"{getattr(args, 'experiment_name', 'loss')}_history.jpg"
    file_path = os.path.join(plot_dir, file_name)
    plt.savefig(file_path, format='jpg')
    
    # Show the plot
    plt.show()
    print(f"Loss history plot saved to {file_path}")



def plot_metric(x, label, plot_dir, args, metric):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    plt.figure(figsize=(10,6))
    plt.plot(x,label=label,color='blue')
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.title(f'{label} over Epochs')
    plt.legend()
    plt.grid(True)

    os.makedirs(plot_dir, exist_ok=True)
    file_name = f"{getattr(args, 'experiment_name', metric)}.jpg"
    file_path =  os.path.join(plot_dir, file_name)
    
    plt.savefig(file_path, format='jpg')
    plt.close()
    
    print(f"Plot saved as {file_path}")

    