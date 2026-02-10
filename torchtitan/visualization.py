import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
import imageio
import io


def visualize_slices(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    filename: str,
    overlay_alpha: float = 0.2,
    fps: int = 10
):
    """
    Creates an animated GIF comparing predictions and targets across a z-stack.
    """
    # 1. Move tensors
    pred_masks = preds.detach().cpu().numpy()
    target_masks = targets.detach().cpu().numpy()
    backgrounds = inputs.numpy()  # inputs.cpu().numpy() 
    
    num_slices = backgrounds.shape[0]

    # 2. Setup colormap
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0])  # Transparent background class
    custom_cmap = ListedColormap(new_colors)
    norm = BoundaryNorm(np.arange(-0.5, num_classes, 1), custom_cmap.N)

    # 3. Setup figure
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

    # 4. Animation update function
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