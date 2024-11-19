
def compute_dice_score(preds, targets, smooth=1e-6):
    '''
    Computes the Dice Score, a measure of similarity between two sets.
    
    Args:
    - preds (torch.Tensor): Predicted segmentation mask (binary or probabilistic tensor).
    - targets (torch.Tensor): Ground truth segmentation mask (binary tensor).
    - smooth (float): Smoothing factor to avoid division by zero.
    
    Formula:
    - Dice Score = 2 * (Intersection) / (Union + smooth)
    
    References:
    - https://oecd.ai/en/catalogue/metrics/dice-score
    - https://en.wikipedia.org/wiki/Dice-Sørensen_coefficient
    
    Returns:
    - float: Dice Score value.
    '''
    
    # Compute the intersection between predictions and targets
    intersection = (preds * targets).sum()
    
    # Compute the union (sum of all values in both predictions and targets)
    union = preds.sum() + targets.sum()
    
    # Calculate the Dice Score using the formula with a smoothing factor to prevent division by zero
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return the Dice Score as a float value
    return dice.item()
