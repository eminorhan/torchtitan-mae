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

def validation_loop(model, val_loader):
    model.eval()
    
    # Stores accumulating 3D logits/probabilities
    # Key: sample_id -> Value: Tensor (Num_Classes, D, H, W)
    predictions = {}
    
    # Stores accumulating 3D Ground Truth labels
    # Key: sample_id -> Value: Tensor (D, H, W)
    ground_truths = {}

    with torch.no_grad():
        for images, labels, metas in val_loader:
            images = images.cuda()
            # labels is shape (Batch, H, W)
            labels = labels.cuda()
            
            outputs = model(images) 
            probs = torch.softmax(outputs, dim=1)
            
            batch_size = images.size(0)
            
            for b in range(batch_size):
                sample_id = metas["sample_id"][b]
                axis = metas["axis"][b].item()
                slice_idx = metas["slice_idx"][b].item()
                vol_shape = tuple(metas["vol_shape"][b].tolist())
                
                # --- Initialize Buffers ---
                if sample_id not in predictions:
                    # 1. Prediction Buffer (Float)
                    num_classes = probs.size(1)
                    predictions[sample_id] = torch.zeros((num_classes,) + vol_shape, device='cuda')
                    
                    # 2. Ground Truth Buffer (Long/Int)
                    # We only need to init this once per sample
                    ground_truths[sample_id] = torch.zeros(vol_shape, dtype=torch.long, device='cuda')

                # --- Accumulate Predictions (All Axes) ---
                current_slice_probs = probs[b] 
                
                if axis == 0:
                    predictions[sample_id][:, slice_idx, :, :] += current_slice_probs
                elif axis == 1:
                    predictions[sample_id][:, :, slice_idx, :] += current_slice_probs
                elif axis == 2:
                    predictions[sample_id][:, :, :, slice_idx] += current_slice_probs
                
                # --- Accumulate Labels (Axis 0 Only) ---
                # We use Axis 0 (Z) to reconstruct the label volume. 
                # We ignore Axis 1 and 2 labels to prevent overwriting/redundancy.
                if axis == 0:
                    ground_truths[sample_id][slice_idx, :, :] = labels[b]

    # 4. Finalize and Compare
    for sample_id, pred_vol in predictions.items():
        # Average the predictions
        final_vol = pred_vol / 3.0 
        final_seg = torch.argmax(final_vol, dim=0) # (D, H, W)
        
        # Retrieve the reconstructed 3D label
        gt_vol = ground_truths[sample_id] # (D, H, W)
        
        # --- Perform Metrics ---
        # Example: Compute Dice/IoU here
        # dice_score = compute_dice(final_seg, gt_vol)
        # print(f"Sample {sample_id} Dice: {dice_score}")

        # --- MEMORY CLEANUP ---
        # Critical if you have many validation samples
        del predictions[sample_id]
        del ground_truths[sample_id]