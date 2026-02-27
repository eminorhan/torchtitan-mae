import os
import zarr
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from torchtitan.datasets import ZarrSegmentationDataset3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- 1. Define Class Mapping ---
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
    Finds the center of the largest connected component for a given label mask.
    Returns (x, y) for plotting text.
    """
    labeled_blobs, num_features = scipy.ndimage.label(label_mask)
    if num_features == 0:
        return None

    # Find the largest blob (ignoring background 0)
    sizes = np.bincount(labeled_blobs.ravel())
    if len(sizes) > 1:
        largest_blob_idx = np.argmax(sizes[1:]) + 1
    else:
        return None

    # Calculate center of mass (returns row, col -> y, x)
    cy, cx = scipy.ndimage.center_of_mass(label_mask, labeled_blobs, largest_blob_idx)
    return cx, cy

def visualize_individual_gifs(
    root_dir: str,
    output_dir: str = "visualizations",
    num_samples: int = 5,
    viz_hw: tuple = (512, 512), # Increased default size for better visibility
    fps: int = 10,
    z_stride: int = 5,   
    overlay_alpha: float = 0.2
):
    """
    Scans the dataset and saves a separate GIF for each sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Scanning for samples in {root_dir}...")
    
    # Initialize dataset
    dataset = ZarrSegmentationDataset3D(root_dir, crop_size=(64, 64, 64), val_crop_size=(64, 64, 64), rank=0, augment=False)
    samples_to_plot = dataset.train_samples[:num_samples]
    
    if len(samples_to_plot) == 0:
        print("No samples found!")
        return

    print(f"Found {len(samples_to_plot)} samples. Processing individually...")

    # --- Setup Colormap (Computed once) ---
    # We estimate max_label_id to be safe (e.g., 255) or use the dictionary length
    max_label_id = max(CLASS_MAPPING.values()) + 1
    colors = plt.cm.get_cmap('gist_ncar', max_label_id)
    new_colors = colors(np.linspace(0, 1, max_label_id))
    new_colors[0, :] = np.array([0, 0, 0, 0]) # Transparent background
    custom_cmap = ListedColormap(new_colors)
    norm = BoundaryNorm(np.arange(-0.5, max_label_id, 1), custom_cmap.N)

    # --- Loop through each sample ---
    for i, sample_info in enumerate(samples_to_plot):
        print(f"[{i+1}/{len(samples_to_plot)}] Loading {sample_info['label_path']}...")
        
        # Load Volume
        raw_vol, label_vol = load_raw_and_label_volume(dataset, sample_info, target_hw=viz_hw, z_stride=z_stride)
        depth = raw_vol.shape[0]
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # No margins
        ax.axis('off')

        # Initial Plot
        im_raw = ax.imshow(raw_vol[0], cmap='gray', vmin=0, vmax=255)
        im_lbl = ax.imshow(label_vol[0], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        
        # Info Text (Top Left)
        info_txt = ax.text(10, 20, f"Sample {i} (z: 0/{depth})", color='white', fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        
        # Container for dynamic label texts
        text_artists = []

        def update(z_idx):
            nonlocal text_artists
            
            # Safe Z index
            safe_z = min(z_idx, depth - 1)
            
            # Update Info Text Box
            info_txt.set_text(f"Sample {i} (z: {safe_z}/{depth})")

            # Update Images
            im_raw.set_data(raw_vol[safe_z])
            im_lbl.set_data(label_vol[safe_z])
            
            # Clear old label texts
            for t in text_artists:
                t.remove()
            text_artists = []

            # Add new label texts
            current_label_slice = label_vol[safe_z]
            unique_ids = np.unique(current_label_slice)
            
            for lbl_id in unique_ids:
                
                name = ID_TO_NAME.get(lbl_id, str(lbl_id))
                center = get_label_center(current_label_slice == lbl_id)
                
                if center:
                    cx, cy = center
                    t = ax.text(cx, cy, name, color='white', fontsize=12, fontweight='bold', ha='center', va='center')
                    t.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
                    text_artists.append(t)
            
            # Return list of artists (not strictly needed if blit=False, but good practice)
            return [im_raw, im_lbl, info_txt] + text_artists

        # Create Animation
        ani = FuncAnimation(fig, update, frames=depth, interval=1000/fps, blit=False)
        
        # Save
        filename = os.path.join(output_dir, f"sample_{i:02d}_viz.gif")
        try:
            ani.save(filename, writer='pillow')
            print(f"  -> Saved: {filename}")
        except Exception as e:
            print(f"  -> Error saving {filename}: {e}")
        
        plt.close(fig) # Close figure to free memory

# --- Example Usage ---
if __name__ == "__main__":
    visualize_individual_gifs(
        root_dir="/lustre/blizzard/stf218/scratch/emin/cellmap-segmentation-challenge/data",
        output_dir="training_crops",
        num_samples=-1,        # Process first k samples
        viz_hw=(512, 512),     # Higher resolution for single plots
        z_stride=1,            # Smaller stride for smoother individual animations
        fps=10
    )