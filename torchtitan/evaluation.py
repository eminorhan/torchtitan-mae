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

def predict_orthoplane_volume(model, volume_tensor, batch_size=16, num_classes=64):
    """
    Performs orthoplane (Z, Y, X) prediction on a 3D volume using distributed inference.
    
    Args:
        model: The 2D segmentation model.
        volume_tensor: Input tensor of shape (D, 3, H, W).
        batch_size: Batch size for the internal 2D slice dataloader.
        num_classes: Number of output classes.
        
    Returns:
        Tensor of shape (D, num_classes, H, W) containing averaged logits/probs.
    """
    device = torch.device('cuda')
    
    # 1. Define the three views (permutations)
    # Z-View: (D, 3, H, W) -> Slices are (3, H, W)
    # Y-View: (H, 3, D, W) -> Slices are (3, D, W)
    # X-View: (W, 3, D, H) -> Slices are (3, D, H)
    views = {
        'Z': {'perm': (0, 1, 2, 3), 'inv_perm': (0, 1, 2, 3)}, 
        'Y': {'perm': (2, 1, 0, 3), 'inv_perm': (2, 1, 0, 3)},
        'X': {'perm': (3, 1, 0, 2), 'inv_perm': (2, 1, 3, 0)} 
    }
    
    accumulated_preds = None

    for axis_name, config in views.items():
        # --- A. Prepare Data for this Axis ---
        # Permute volume to place the slicing axis at dim 0 (Batch)
        view_data = volume_tensor.permute(*config['perm']).contiguous()
        original_size = view_data.size(0)
        
        # Create a proper DataLoader for this axis
        dataset = TensorDataset(view_data)
        
        # DistributedSampler ensures each GPU processes a subset of slices
        sampler = DistributedSampler(
            dataset, 
            shuffle=False,  # Order matters for reconstruction
            drop_last=False
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

        # --- B. Distributed Inference Loop ---
        local_preds = []
        model.eval()
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device) # Unpack TensorDataset
                outputs = model(inputs)
                local_preds.append(outputs)
        
        # Concatenate local predictions: (Local_Slices, Classes, H_view, W_view)
        if local_preds:
            local_tensor = torch.cat(local_preds, dim=0)
        else:
            # Handle edge case where a rank gets 0 samples
            # Construct a dummy tensor with correct shape but 0 batch dim
            spatial = view_data.shape[2:]
            local_tensor = torch.zeros((0, num_classes, *spatial), device=device)

        # --- C. Gather Results from all Ranks ---
        # We need to gather the predictions to reconstruct the full volume.
        # Note: all_gather requires tensors to be same size across ranks, 
        # but DistributedSampler might pad. A robust gather handles this, 
        # but for simplicity we assume standard setup or truncate padding.
        
        # 1. Gather sizes to handle uneven batches
        local_size = torch.tensor([local_tensor.size(0)], device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(all_sizes, local_size)
        
        # 2. Pad local tensor to max size for safe all_gather
        max_size = max([s.item() for s in all_sizes])
        pad_amount = max_size - local_tensor.size(0)
        if pad_amount > 0:
            padded_tensor = torch.nn.functional.pad(local_tensor, (0,0, 0,0, 0,0, 0,pad_amount))
        else:
            padded_tensor = local_tensor

        # 3. Gather
        gathered_list = [torch.zeros_like(padded_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_list, padded_tensor)

        # 4. Remove padding and Concatenate
        clean_list = []
        for i, size_tensor in enumerate(all_sizes):
            clean_list.append(gathered_list[i][:size_tensor.item()])
        
        # Full prediction for this axis: (Total_Slices, Classes, H_view, W_view)
        full_axis_pred = torch.cat(clean_list, dim=0)

        # 5. Handle DistributedSampler Duplication
        # DistributedSampler adds padding to make total count divisible by world_size.
        # We must truncate to the original volume size.
        full_axis_pred = full_axis_pred[:original_size]

        # --- D. Permute Back and Accumulate ---
        # Permute back to (D, Classes, H, W)
        # Note: We added a Class dimension at dim 1, so indices shift by 1 compared to input
        # Input perm was (Batch, C, H, W). Output is (Batch, Classes, H, W).
        # We need to apply the inverse perm logic carefully.
        
        if axis_name == 'Z':
            # Current: (D, Cls, H, W). Target: (D, Cls, H, W)
            aligned_pred = full_axis_pred 
        elif axis_name == 'Y':
            # Current: (H, Cls, D, W). Target: (D, Cls, H, W)
            aligned_pred = full_axis_pred.permute(2, 1, 0, 3) 
        elif axis_name == 'X':
            # Current: (W, Cls, D, H). Target: (D, Cls, H, W)
            aligned_pred = full_axis_pred.permute(2, 1, 3, 0)

        if accumulated_preds is None:
            accumulated_preds = aligned_pred
        else:
            accumulated_preds += aligned_pred

    # Average the 3 views
    final_preds = accumulated_preds / 3.0
    
    return final_preds