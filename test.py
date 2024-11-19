import os   ######## google colabde a100 gpusu ile batch size 16, epoch sayısı 20 
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import test_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score

def test_model(model, args, save_path):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - args (argparse.Namespace): Parsed arguments for device, batch size, etc.
    - save_path (str): Directory where results (e.g., metrics plot) will be saved.
    
    Functionality:
    - Sets the model to evaluation mode and iterates over the test dataset.
    - Computes the Dice score for each batch and calculates the average.
    - Saves a plot of the Dice coefficient history.
    '''
    model.to(args.device)
    model.eval()
    dice_scores = []
    os.makedirs(save_path,exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            images, masks = batch
            images, masks = images.to(args.device), masks.to(args.device)
            outputs = model(images)
            dice_score = compute_dice_score(outputs, masks)
            dice_scores.append(dice_score)

            if batch_idx <3:
                visualize_predictions(images=images.cpu(),masks=masks.cpu(),
                    outputs=outputs.cpu(),save_path=save_path,epoch=0,  batch_idx=batch_idx)
                
    avg_dice_score = torch.tensor(dice_scores).mean().item()
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")

    plot_metric(x=dice_scores, label="Dice Coefficient", plot_dir=save_path, args=args, metric="dice_score")

    dice_file_path = os.path.join(save_path, "dice_scores.txt")
    with open(dice_file_path, "w") as f:
        for score in dice_scores:
            f.write(f"{score:.4f}\n")
    print(f"Dice scores saved to {dice_file_path}")

if __name__ == '__main__':

    args = test_arg_parser()
    save_path = "/results"
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="/madison-stomach/madison-stomach", 
                            mode="test")

    test_dataloader = DataLoader(dataset, batch_size=args.bs)

    # Define and load your model
    model =  UNet(in_channels=1, out_channels=1)
    model = torch.load(args.model_path, map_location=args.device)

    test_model(model=model,
                args=args,
                save_path=save_path)