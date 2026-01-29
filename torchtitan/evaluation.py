import torch

def compute_pixel_accuracy(logits, targets):
    """
    Computes the percentage of correctly classified pixels.
    Args:
        logits: Model output tensor (B, C, H, W) or (B, C, D, H, W)
        targets: Ground truth tensor (B, H, W) or (B, D, H, W)
    """
    # Convert logits to class predictions
    preds = torch.argmax(logits, dim=1)
    
    # Count correct pixels
    correct = (preds == targets).sum().item()
    total = targets.numel()
    
    return correct / total

def compute_confusion_matrix(preds, targets, num_classes, ignore_index=None):
    """
    Computes the confusion matrix for mIoU calculation.
    Args:
        preds: (B, C, H, W) or (B, C, D, H, W) - Raw logits or softmax output
        targets: (B, H, W) or (B, D, H, W) - Ground truth labels
        num_classes: int
        ignore_index: int (optional) - Label to ignore (e.g. 255 for void)
    Returns:
        conf_matrix: (num_classes, num_classes) tensor on the same device
    """
    # Get class predictions
    preds = torch.argmax(preds, dim=1)  # Shape: (B, H, W) or (B, D, H, W)
    
    # Flatten inputs to 1D lists of pixels
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Filter out 'ignore_index' if necessary
    if ignore_index is not None:
        mask = (targets_flat != ignore_index)
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]
    
    # Compute confusion matrix using bincount
    unique_mapping = targets_flat * num_classes + preds_flat
    
    # Count occurrences
    hist = torch.bincount(unique_mapping, minlength=num_classes**2)
    
    # Reshape to (num_classes, num_classes)
    conf_matrix = hist.reshape(num_classes, num_classes)
    
    return conf_matrix.float()
