import os
import zarr
import numpy as np
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
from torchtitan.datasets import ZarrSegmentationDataset3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

def load_raw_and_label_volume(dataset, sample_info, target_hw=(256, 256), z_stride=1):
    """
    Helper to load a single sample's full 3D volume (Raw + Label).
    Reuses the logic from the validation_iterator but returns numpy arrays.
    """
    zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
    
    # Load full 3D Label
    label_path = sample_info['label_path']
    label_array_3d = zarr_root[label_path][:] # Load entire volume into memory
    
    # Metadata for Label
    label_attrs_group_path = os.path.dirname(label_path)
    label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
    label_scale_name = os.path.basename(label_path)
    label_scale, label_translation = dataset._parse_ome_ngff_metadata(label_attrs, label_scale_name)
    
    if label_scale is None: 
        label_scale, label_translation = [1.0]*3, [0.0]*3

    # Determine visualization target shape
    # We keep the original Z depth, but resize H/W to target_hw for grid uniformity
    d_dim = label_array_3d.shape[0]
    viz_shape = (d_dim, target_hw[0], target_hw[1])

    # Resize Label if needed (Nearest Neighbor)
    zoom_factors = [t/s for t, s in zip(viz_shape, label_array_3d.shape)]
    if label_array_3d.shape != viz_shape:
        label_vol = scipy.ndimage.zoom(label_array_3d, zoom_factors, order=0, prefilter=False)
    else:
        label_vol = label_array_3d

    # Find and Load Raw Data
    raw_group_path = sample_info['raw_path_group']
    raw_attrs = zarr_root[raw_group_path].attrs.asdict()
    best_raw_path, raw_scale, raw_translation = dataset._find_best_raw_scale(label_scale, raw_attrs)
    
    raw_array_full = zarr_root[os.path.join(raw_group_path, best_raw_path)]
    
    # Calculate physical crop coordinates
    # (Mapping physical label coordinates to raw voxel coordinates)
    label_phys_size = [s * sc for s, sc in zip(label_array_3d.shape, label_scale)]
    
    rel_start = [ls - rs for ls, rs in zip(label_translation, raw_translation)]
    start_raw = [int(round(p / s)) for p, s in zip(rel_start, raw_scale)]
    
    rel_end_phys = [s + sz for s, sz in zip(rel_start, label_phys_size)]
    end_raw = [int(round(p / s)) for p, s in zip(rel_end_phys, raw_scale)]

    # Safe slicing
    raw_shape_full = raw_array_full.shape
    slices = [slice(max(0, start_raw[i]), min(raw_shape_full[i], end_raw[i])) for i in range(3)]
    
    raw_crop = raw_array_full[tuple(slices)]
    
    # Handle empty crop edge case
    if any(s == 0 for s in raw_crop.shape):
        raw_crop = np.zeros(viz_shape, dtype=raw_array_full.dtype)

    # Resize Raw to match the Viz Shape (Linear Interpolation)
    raw_zoom = [t/s for t, s in zip(viz_shape, raw_crop.shape)]
    if raw_crop.shape != viz_shape:
        raw_vol = scipy.ndimage.zoom(raw_crop, raw_zoom, order=1, prefilter=False)
    else:
        raw_vol = raw_crop

    # Apply Z-Stride (Subsampling)
    # We slice [::z_stride] to skip frames
    raw_vol = raw_vol[::z_stride]
    label_vol = label_vol[::z_stride]

    return raw_vol, label_vol


def visualize_training_grid(
    root_dir: str,
    grid_size: tuple = (4, 4),
    filename: str = "training_grid_animation.gif",
    viz_hw: tuple = (256, 256),
    fps: int = 10,
    z_stride: int = 10,   
    overlay_alpha: float = 0.2
):
    """
    Scans the dataset, loads the first N=grid_size samples, and plots a synchronized animation.
    """
    print(f"Scanning for samples in {root_dir}...")
    
    # Initialize dataset just to parse file lists (dummy crop_size)
    dataset = ZarrSegmentationDataset3D(root_dir, crop_size=(64, 64, 64), rank=0, augment=False)
    
    # Pick the first N samples
    num_samples = grid_size[0] * grid_size[1]
    samples_to_plot = dataset.train_samples[:num_samples]
    
    if len(samples_to_plot) == 0:
        print("No samples found!")
        return

    print(f"Loading {len(samples_to_plot)} samples (stride={z_stride})...")
    volumes = []
    max_z = 0

    for i, sample_info in enumerate(samples_to_plot):
        print(f"  Loading sample {i+1}/{len(samples_to_plot)}: {sample_info['label_path']}...")
        raw, label = load_raw_and_label_volume(dataset, sample_info, target_hw=viz_hw, z_stride=z_stride)
        volumes.append({'raw': raw, 'label': label})
        if raw.shape[0] > max_z:
            max_z = raw.shape[0]

    # --- Setup Visualization ---
    print("Preparing animation...")
    
    # Define Colormap (0=Transparent, others=Random Bright Colors)
    # Assuming max label ID is 255 for uint8 labels, adjust if necessary
    max_label_id = max([v['label'].max() for v in volumes]) if volumes else 1
    num_classes = int(max_label_id) + 1
    
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0]) # Make background transparent
    custom_cmap = ListedColormap(new_colors)
    # Norm is critical so integer labels map exactly to specific colors
    norm = BoundaryNorm(np.arange(-0.5, num_classes, 1), custom_cmap.N)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(4 * grid_size[1], 4 * grid_size[0]))

    # This sets the margins to 1% of the figure width/height and reduces the space between subplots to 1%.
    # Top is set to 0.93 to leave just enough room for the suptitle.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.01, hspace=0.01)
    # ------------------------------

    axes = np.array(axes).flatten() # Flatten to easy 1D index

    plot_objects = []

    # Initialize the first frame
    for i, ax in enumerate(axes):
        if i < len(volumes):
            vol = volumes[i]
            # Plot Raw (Grayscale)
            im_raw = ax.imshow(vol['raw'][0], cmap='gray', vmin=0, vmax=255)
            # Plot Label (Overlay)
            im_lbl = ax.imshow(vol['label'][0], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
            
            # Add text annotation
            txt = ax.text(5, 15, f"Crop {i}", color='white', fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
            plot_objects.append({'raw': im_raw, 'lbl': im_lbl, 'txt': txt})
        
        ax.axis('off')

    title_text = fig.suptitle(f"Z-slice (stride {z_stride}): 0/{max_z}", fontsize=16)

    def update(z_idx):
        # Update title
        title_text.set_text(f"Z-slice (stride {z_stride}): {z_idx}/{max_z}")
        
        artists = [title_text]
        
        for i, obj in enumerate(plot_objects):
            vol = volumes[i]
            current_z_depth = vol['raw'].shape[0]
            
            # If the volume is shallower than current z_idx, just hold the last frame
            # (or you could blank it out)
            safe_z = min(z_idx, current_z_depth - 1)
            
            obj['raw'].set_data(vol['raw'][safe_z])
            obj['lbl'].set_data(vol['label'][safe_z])
            artists.extend([obj['raw'], obj['lbl']])
            
        return artists

    ani = FuncAnimation(fig, update, frames=max_z, interval=1000/fps, blit=False) # blit=False is often more stable for subplots
    
    try:
        ani.save(filename, writer='pillow')
        print(f"Success! Saved animation grid to {filename}")
    except Exception as e:
        print(f"Error saving animation (check if pillow/imagemagick is installed): {e}")
    finally:
        plt.close(fig)

# --- Example Usage ---
visualize_training_grid(
    root_dir="/lustre/blizzard/stf218/scratch/emin/cellmap-segmentation-challenge/data",
    grid_size=(6, 6),
    filename="training_crops.gif"
)