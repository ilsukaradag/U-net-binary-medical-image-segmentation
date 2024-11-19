import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from model.unet import UNet
from utils.model_utils import train_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score



def train_model(model, train_loader, val_loader, optimizer, criterion, args, save_path):
    '''
    Trains the given model over multiple epochs, tracks training and validation losses, 
    and saves model checkpoints periodically.

    Args:
    - model (torch.nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration (e.g., epochs, batch size, device).
    - save_path (str): Directory path to save model checkpoints and training history.

    Functionality:
    - Creates directories to save results and checkpoints.
    - Calls `train_one_epoch` to train and validate the model for each epoch.
    - Saves model checkpoints every 5 epochs.
    - Plots the training and validation loss curves and the Dice coefficient curve.
    '''
    DEVICE = torch.device(args.device)
    BATCH_SIZE = args.bs
    EPOCH = args.epoch
    LR = args.lr
    EXP_ID = args.exp_id
    os.makedirs(os.path.join(save_path), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []

    for epoch in range(args.epoch):
        train_one_epoch(model, 
                        train_loader, 
                        val_loader, 
                        train_loss_history, 
                        val_loss_history, 
                        dice_coef_history, 
                        optimizer, 
                        criterion, 

                        args, 
                        
                        epoch, 
                        save_path)
        
        
        

    plot_train_val_history(train_loss_history, val_loss_history, save_path, args)
    plot_metric(dice_coef_history, label="dice coeff", plot_dir=save_path, args=args, metric='dice_coeff')

def train_one_epoch(model, train_loader, val_loader, train_loss_history, val_loss_history, 
                    dice_coef_history, optimizer, criterion, args, epoch, save_path):
    '''
    Performs one full epoch of training and validation, computes metrics, and visualizes predictions.

    Args:
    - model (torch.nn.Module): The neural network model to train.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - train_loss_history (list): List to store the average training loss per epoch.
    - val_loss_history (list): List to store the average validation loss per epoch.
    - dice_coef_history (list): List to store the Dice coefficient per epoch.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration.
    - epoch (int): The current epoch number.
    - save_path (str): Directory path to save visualizations and model checkpoints.

    Functionality:
    - Sets the model to training mode and performs a forward and backward pass for each batch in the training data.
    - Computes the training loss and updates the weights.
    - Sets the model to evaluation mode and computes validation loss and Dice coefficients.
    - Visualizes predictions periodically and saves them to the specified directory.
    - Appends the average training and validation losses, and the Dice coefficient to their respective lists.
    - Prints the Dice coefficient and loss values for the current epoch.
    '''

    # Training phase
    DEVICE = torch.device(args.device)
    model.train()
    running_loss = 0.0
    train_f1 = 0.0
    all_preds_train = []
    all_masks_train = []
    running_f1 = 0.0
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=True)
    batch_count = 0

    for images, masks in train_bar: #cchange to trainbar
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        batch_count +=1
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds_train.append(preds.cpu())  # Keep tensors on GPU for now
        all_masks_train.append(masks.cpu())

        if batch_count % 5 == 0:  # e.g., visualize every 5 batches
            visualize_predictions(images.cpu(), preds.cpu(), masks.cpu(), save_path, batch_count, 'train')
        if batch_count % 100 == 0:
            torch.save(model, os.path.join(save_path,'model', f'unet_epoch{epoch}_batch{batch_count}.pt'))

    all_preds_train = torch.cat(all_preds_train).cpu().numpy().flatten()
    all_masks_train = torch.cat(all_masks_train).cpu().numpy().flatten()

    all_preds_train = (all_preds_train > 0.5).astype(np.int32)  # Binarize predictions
    all_masks_train = (all_masks_train > 0.5).astype(np.int32)
    train_f1 = f1_score(all_masks_train, all_preds_train, zero_division=1)  # F1 score on CPU

    train_bar.set_postfix({'Loss': running_loss / batch_count, 'F1': train_f1})
    train_dice_score = f1_score(all_masks_train, all_preds_train, zero_division=1)

    train_loss_history.append(running_loss / len(train_loader))
    dice_coef_history.append(train_dice_score)

    print(f"Epoch [{epoch+1}/{args.epoch}], Training Loss: {train_loss_history[-1]:.4f}, Training Dice Score: {train_dice_score:.4f}")
    
    model.eval()
    val_running_loss = 0.0
    all_preds_val = []
    all_masks_val = []
    running_f1 = 0.0

    with torch.no_grad():
        for images, masks  in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', leave=False):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds_val.append(preds.cpu())  # Keep tensors on GPU for now
            all_masks_val.append(masks.cpu())


        # Incremental F1 computation
            visualize_predictions(images.cpu(), preds.cpu(), masks.cpu(), save_path, epoch, 'val')
            

   
    all_preds_val = torch.cat(all_preds_val).cpu().numpy().flatten()
    all_masks_val = torch.cat(all_masks_val).cpu().numpy().flatten()

    all_preds_val = (all_preds_val > 0.5).astype(np.int32)  # Binarize predictions
    all_masks_val = (all_masks_val > 0.5).astype(np.int32)
    val_dice_score = f1_score(all_masks_val, all_preds_val, zero_division=1)

    val_loss_history.append(val_running_loss / len(val_loader))
    try:
      print("model being saved")
      torch.save(model.state_dict(), os.path.join(save_path, 'model', f'unet_{epoch}.pt'))
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")

    print(f"Epoch [{epoch+1}/{args.epoch}], Validation Loss: {val_loss_history[-1]:.4f}, Validation Dice Score: {val_dice_score:.4f}")



if __name__ == '__main__':

    args = train_arg_parser()
    print("Current Working Directory:", os.getcwd())
    save_path = os.path.join("./results")
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="./madison-stomach/madison-stomach", 
                            mode=args.mode)

    # Define train and val indices
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    # Define Subsets of to create trian and validation dataset
    train_subset = Subset(dataset, train_indices)#Subset(dataset, train_indices)
    val_subset =Subset(dataset, val_indices)#Subset(train_dataset, val_indices)

    # Define dataloader
    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)#DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)#DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    DEVICE = torch.device(args.device)
    # Define your model
    model = UNet()
    model = model.to(DEVICE)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                args=args,
                save_path=save_path)
