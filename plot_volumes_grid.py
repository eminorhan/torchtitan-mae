import os
import zarr
import numpy as np
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # for text outlines
from torchtitan.datasets import ZarrSegmentationDataset3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

# Class-index mapping
CLASS_MAPPING = {
    'ecs': 0, 'pm': 1, 'mito_mem': 2, 'mito_lum': 3, 'mito_ribo': 4, 'golgi_mem': 5, 
    'golgi_lum': 6, 'ves_mem': 7, 'ves_lum': 8, 'endo_mem': 9, 'endo_lum': 10, 
    'lyso_mem': 11, 'lyso_lum': 12, 'ld_mem': 13, 'ld_lum': 14, 'er_mem': 15, 
    'er_lum': 16, 'eres_mem': 17, 'eres_lum': 18, 'ne_mem': 19, 'ne_lum': 20, 
    'np_out': 21, 'np_in': 22, 'hchrom': 23, 'nhchrom': 24, 'echrom': 25, 
    'nechrom': 26, 'nucpl': 27, 'nucleo': 28, 'mt_out': 29, 'cent': 30, 
    'cent_dapp': 31, 'cent_sdapp': 32, 'ribo': 33, 'cyto': 34, 'mt_in': 35, 
    'nuc': 36, 'vim': 37, 'glyco': 38, 'golgi': 39, 'ves': 40, 'endo': 41, 
    'lyso': 42, 'ld': 43, 'rbc': 44, 'eres': 45, 'perox_mem': 46, 'perox_lum': 47, 
    'perox': 48, 'mito': 49, 'er': 50, 'ne': 51, 'np': 52, 'chrom': 53, 
    'mt': 54, 'isg_mem': 55, 'isg_lum': 56, 'isg_ins': 57, 'isg': 58, 'cell': 59, 
    'actin': 60, 'tbar': 61, 'bm': 62, 'er_mem_all': 63, 'ne_mem_all': 64, 
    'cent_all': 65, 'chlor_mem': 66, 'chlor_lum': 67, 'chlor_sg': 68, 
    'chlor': 69, 'vac_mem': 70, 'vac_lum': 71, 'vac': 72, 'pd': 73
}

# Create inverse mapping: ID -> Name
ID_TO_NAME = {v: k for k, v in CLASS_MAPPING.items()}

def load_raw_and_label_volume(dataset, sample_info, target_hw=(256, 256), z_stride=1):
    """
    Helper to load a single sample's full 3D volume (Raw + Label).
    """
    zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
    
    # Load full 3D Label
    label_path = sample_info['label_path']
    label_array_3d = zarr_root[label_path][:] 
    
    # Metadata for Label
    label_attrs_group_path = os.path.dirname(label_path)
    label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
    label_scale_name = os.path.basename(label_path)
    label_scale, label_translation = dataset._parse_ome_ngff_metadata(label_attrs, label_scale_name)
    
    if label_scale is None: 
        label_scale, label_translation = [1.0]*3, [0.0]*3

    # Determine visualization target shape
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
    label_phys_size = [s * sc for s, sc in zip(label_array_3d.shape, label_scale)]
    
    rel_start = [ls - rs for ls, rs in zip(label_translation, raw_translation)]
    start_raw = [int(round(p / s)) for p, s in zip(rel_start, raw_scale)]
    
    rel_end_phys = [s + sz for s, sz in zip(rel_start, label_phys_size)]
    end_raw = [int(round(p / s)) for p, s in zip(rel_end_phys, raw_scale)]

    # Safe slicing
    raw_shape_full = raw_array_full.shape
    slices = [slice(max(0, start_raw[i]), min(raw_shape_full[i], end_raw[i])) for i in range(3)]
    
    raw_crop = raw_array_full[tuple(slices)]
    
    if any(s == 0 for s in raw_crop.shape):
        raw_crop = np.zeros(viz_shape, dtype=raw_array_full.dtype)

    # Resize Raw to match the Viz Shape (Linear Interpolation)
    raw_zoom = [t/s for t, s in zip(viz_shape, raw_crop.shape)]
    if raw_crop.shape != viz_shape:
        raw_vol = scipy.ndimage.zoom(raw_crop, raw_zoom, order=1, prefilter=False)
    else:
        raw_vol = raw_crop

    # Apply Z-Stride
    raw_vol = raw_vol[::z_stride]
    label_vol = label_vol[::z_stride]

    return raw_vol, label_vol


def get_label_center(label_mask):
    """
    Finds a representative coordinate for a label to place the text.
    It identifies the largest connected component and returns its center of mass.
    This prevents the label from appearing in empty space between two distant blobs.
    """
    # 1. Label connected components
    labeled_blobs, num_features = scipy.ndimage.label(label_mask)
    
    if num_features == 0:
        return None

    # 2. Find the largest blob (by area)
    # bincount is fast for counting pixels per label
    sizes = np.bincount(labeled_blobs.ravel())
    # sizes[0] is background (0), so ignore it.
    if len(sizes) > 1:
        largest_blob_idx = np.argmax(sizes[1:]) + 1
    else:
        return None

    # 3. Calculate center of mass for that specific blob
    # center_of_mass returns (y, x)
    cy, cx = scipy.ndimage.center_of_mass(label_mask, labeled_blobs, largest_blob_idx)
    return cx, cy


def visualize_training_grid(
    root_dir: str,
    grid_size: tuple = (4, 4),
    filename: str = "training_grid_animation.gif",
    viz_hw: tuple = (256, 256),
    fps: int = 10,
    z_stride: int = 2,   
    overlay_alpha: float = 0.2
):
    print(f"Scanning for samples in {root_dir}...")
    
    dataset = ZarrSegmentationDataset3D(root_dir, crop_size=(64, 64, 64), val_crop_size=(64, 64, 64), rank=0, augment=False)
    
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
    
    # Define Colormap
    max_label_id = max([v['label'].max() for v in volumes]) if volumes else 1
    num_classes = int(max_label_id) + 1
    
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0]) # Transparent background
    custom_cmap = ListedColormap(new_colors)
    norm = BoundaryNorm(np.arange(-0.5, num_classes, 1), custom_cmap.N)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(4 * grid_size[1], 4 * grid_size[0]))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.01, hspace=0.01)

    axes = np.array(axes).flatten()

    plot_objects = []

    # Initialize subplots
    for i, ax in enumerate(axes):
        if i < len(volumes):
            vol = volumes[i]
            # Raw Image
            im_raw = ax.imshow(vol['raw'][0], cmap='gray', vmin=0, vmax=255)
            # Label Overlay
            im_lbl = ax.imshow(vol['label'][0], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
            # # Static Title
            # txt = ax.text(5, 15, f"Crop {i}", color='white', fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
            # Store objects. 'dynamic_texts' holds the class labels that change every frame
            plot_objects.append({
                'raw': im_raw, 
                'lbl': im_lbl, 
                'dynamic_texts': [], # List to track text artists for removal
                'ax': ax # Reference to ax to add new text
            })
        
        ax.axis('off')

    title_text = fig.suptitle(f"Z-slice (stride {z_stride}): 0/{max_z}", fontsize=16)

    def update(z_idx):
        title_text.set_text(f"Z-slice (stride {z_stride}): {z_idx}/{max_z}")
        artists = [title_text]
        
        for i, obj in enumerate(plot_objects):
            vol = volumes[i]
            current_z_depth = vol['raw'].shape[0]
            safe_z = min(z_idx, current_z_depth - 1)
            
            # 1. Update Image Data
            obj['raw'].set_data(vol['raw'][safe_z])
            obj['lbl'].set_data(vol['label'][safe_z])
            artists.extend([obj['raw'], obj['lbl']])

            # 2. Update Dynamic Class Labels
            # First, remove old text from previous frame
            for text_artist in obj['dynamic_texts']:
                text_artist.remove()
            obj['dynamic_texts'] = [] # Clear list

            # Get current slice data
            current_label_slice = vol['label'][safe_z]
            unique_ids = np.unique(current_label_slice)

            # Loop through unique labels in this slice
            for lbl_id in unique_ids:
                # if lbl_id == 0: continue # Skip background
                
                # Get the name
                name = ID_TO_NAME.get(lbl_id, str(lbl_id))
                
                # Find "Single Example Region" (Center of largest blob)
                center = get_label_center(current_label_slice == lbl_id)
                
                if center:
                    cx, cy = center
                    # Add text
                    t = obj['ax'].text(
                        cx, cy, name, 
                        color='white', 
                        fontsize=12, 
                        fontweight='bold', 
                        ha='center', 
                        va='center'
                    )
                    # Add outline for visibility against colored segments
                    t.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
                    
                    obj['dynamic_texts'].append(t)
                    artists.append(t)
            
        return artists

    ani = FuncAnimation(fig, update, frames=max_z, interval=1000/fps, blit=False)
    
    try:
        ani.save(filename, writer='pillow')
        print(f"Success! Saved animation grid to {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close(fig)

# --- Example Usage ---
if __name__ == "__main__":
    visualize_training_grid(
        root_dir="/lustre/blizzard/stf218/scratch/emin/cellmap-segmentation-challenge/data",
        grid_size=(4, 4),
        filename="training_crops_labeled.gif"
    )