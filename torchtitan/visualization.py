import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
import imageio
import io


def visualize_slices_3d(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    step: int,
    sample_idx: int = 0,
    overlay_alpha: float = 0.3,
    fps: int = 10
):
    """
    Visualizes slices from a 3D volume as a side-by-side GIF animation of predictions and ground truth.

    Args:
        inputs (torch.Tensor): The input volume (B, C, D, H, W). Assumes first channel is image data.
        preds (torch.Tensor): The model output logits (B, num_classes, D, H, W).
        targets (torch.Tensor): The ground truth labels (B, 1, D, H, W).
        num_classes (int): The total number of segmentation classes.
        step (int): Step number, used for saving the output file.
        sample_idx (int): The index of the sample in the batch to visualize.
        overlay_alpha (float): Transparency of the mask overlay (0.1 is faint, 0.5 is half-opaque).
        fps (int): Frames per second for the output GIF.
    """
    # Convert logits and select sample 
    pred_masks = torch.argmax(preds, dim=1)  # Shape: (B, D, H, W)

    input_sample = inputs[sample_idx]
    pred_mask_sample = pred_masks[sample_idx]
    target_mask_sample = targets[sample_idx]

    # Move tensors to CPU and numpy
    input_image = input_sample[0].cpu().numpy()
    pred_mask = pred_mask_sample.cpu().numpy()
    # Squeeze the channel dimension (1) from the target mask
    target_mask = target_mask_sample.squeeze(0).cpu().numpy() # Shape (D, H, W)

    # Create a consistent colormap
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0])  # Set background class (index 0) to transparent
    custom_cmap = ListedColormap(new_colors)
    bounds = np.arange(-0.5, num_classes, 1)
    norm = BoundaryNorm(bounds, custom_cmap.N)

    # Generate frames for the GIF
    frames = []
    depth = input_image.shape[0]

    for slice_idx in range(0, depth, 4):  # plot every k-th slice
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # fig.suptitle(f"Slice {slice_idx + 1} / {depth}", fontsize=14)

        # Left Plot: Predictions
        ax = axes[0]
        ax.imshow(input_image[slice_idx], cmap='gray')
        ax.imshow(pred_mask[slice_idx], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        ax.set_title("Prediction")
        ax.axis('off')

        # Right Plot: Ground truth
        ax = axes[1]
        ax.imshow(input_image[slice_idx], cmap='gray')
        ax.imshow(target_mask[slice_idx], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        ax.set_title("Ground truth")
        ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        # Convert the matplotlib plot to an image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        
        # Close the figure to free up memory
        plt.close(fig)

    # Save the frames as a GIF animation
    output_filename = f"step_{step}_3d_animation.gif"
    imageio.mimsave(output_filename, frames, fps=fps)
    print(f"Saved animation to {output_filename}")


def visualize_slices_2d(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    filename: str = "validation_stack.gif",
    overlay_alpha: float = 0.2,
    fps: int = 10
):
    """
    Creates an animated GIF comparing predictions and targets across a z-stack.
    """
    # 1. Pre-process Tensors
    # Convert logits (B, C, H, W) -> labels (B, H, W)
    pred_masks = torch.argmax(preds, dim=1).cpu().numpy()
    target_masks = targets.cpu().numpy()
    # Take only the first channel for the grayscale background (B, H, W)
    backgrounds = inputs[:, 0].cpu().numpy() 
    
    num_slices = backgrounds.shape[0]

    # 2. Setup Colormap (Same as your 2D function)
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0])  # Transparent background class
    custom_cmap = ListedColormap(new_colors)
    norm = BoundaryNorm(np.arange(-0.5, num_classes, 1), custom_cmap.N)

    # 3. Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initialize plot objects for updating in the animation loop
    # We use empty arrays initially
    im_bg_left = axes[0].imshow(backgrounds[0], cmap='gray')
    im_mask_left = axes[0].imshow(pred_masks[0], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
    axes[0].set_title("Prediction")
    axes[0].axis('off')

    im_bg_right = axes[1].imshow(backgrounds[0], cmap='gray')
    im_mask_right = axes[1].imshow(target_masks[0], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    slice_text = fig.text(0.5, 0.02, f"Slice 0/{num_slices}", ha='center', fontsize=12)

    # 4. Animation Update Function
    def update(i):
        # Update backgrounds
        im_bg_left.set_data(backgrounds[i])
        im_bg_right.set_data(backgrounds[i])
        
        # Update overlay masks
        im_mask_left.set_data(pred_masks[i])
        im_mask_right.set_data(target_masks[i])
        
        slice_text.set_text(f"Slice {i+1}/{num_slices}")
        return [im_bg_left, im_bg_right, im_mask_left, im_mask_right, slice_text]

    # 5. Create and Save Animation
    ani = FuncAnimation(fig, update, frames=num_slices, interval=1000/fps, blit=True)
    
    # Note: Requires 'pillow' or 'imagemagick' installed: pip install pillow
    ani.save(filename, writer='pillow')
    plt.close(fig)
    print(f"Saved animation to {filename}")


# def visualize_slices_2d(
#     inputs: torch.Tensor,
#     preds: torch.Tensor,
#     targets: torch.Tensor,
#     num_classes: int,
#     step: int,
#     overlay_alpha: float = 0.3
# ):
#     """
#     Visualizes all 2D images in a batch with their predicted and ground truth masks.

#     Args:
#         inputs (torch.Tensor): The input image tensor (B, C, H, W). Assumes the first channel is the image data.
#         preds (torch.Tensor): The model output logits (B, num_classes, H, W).
#         targets (torch.Tensor): The ground truth labels (B, 1, H, W).
#         num_classes (int): The total number of segmentation classes.
#     """
#     # Get batch size
#     batch_size = inputs.shape[0]

#     # Convert prediction logits to discrete class labels
#     pred_masks = torch.argmax(preds, dim=1)  # Shape: (B, H, W)

#     # Create a consistent colormap for all classes
#     colors = plt.cm.get_cmap('gist_ncar', num_classes)
#     new_colors = colors(np.linspace(0, 1, num_classes))
#     new_colors[0, :] = np.array([0, 0, 0, 0])
#     custom_cmap = ListedColormap(new_colors)
#     bounds = np.arange(-0.5, num_classes, 1)
#     norm = BoundaryNorm(bounds, custom_cmap.N)

#     # Set up the plot for the entire batch. `squeeze=False` ensures axes is always 2D.
#     fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8.5), squeeze=False)

#     # Loop through each sample in the batch
#     for i in range(batch_size):
#         # Prepare data for the i-th sample
#         input_image = inputs[i, 0].cpu().numpy()
#         pred_mask = pred_masks[i].cpu().numpy()
#         target_mask = targets[i].cpu().numpy() # Extract from channel dim

#         # Top Row: Prediction
#         ax = axes[0, i]
#         ax.imshow(input_image, cmap='gray')
#         ax.imshow(pred_mask, cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
#         ax.set_title(f"Prediction (Sample {i})")
#         ax.axis('off')

#         # Bottom Row: Ground truth
#         ax = axes[1, i]
#         ax.imshow(input_image, cmap='gray')
#         ax.imshow(target_mask, cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
#         ax.set_title(f"Ground truth (Sample {i})")
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(f"step_{step}_2d.jpeg", bbox_inches='tight')
#     plt.close(fig)
