import os
import re
import zarr
import numpy as np
from glob import glob
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value

def get_recon_sort_key(recon_name):
    """Extracts the integer from recon names (e.g. 'recon-2') for natural sorting."""
    match = re.search(r'recon-(\d+)', recon_name)
    return int(match.group(1)) if match else float('inf')

def get_em_subfolder_sort_key(folder_name):
    """Prioritizes by data type suffix regardless of the prefix (fibsem, tem, etc.)."""
    if folder_name.endswith('-uint8'):
        return (0, folder_name)
    elif folder_name.endswith('-uint8_1'):
        return (1, folder_name)
    elif folder_name.endswith('-uint16'):
        return (2, folder_name)
    elif folder_name.endswith('-int16'):
        return (3, folder_name)
    else:
        # Fallback priority for unknown types, then sort alphabetically
        return (4, folder_name)

def generate_2d_slices(root_dir):
    """Generator yielding 2D slices by loading the entire volume into RAM."""
    zarr_paths = glob(os.path.join(root_dir, '*/*.zarr'))
    axis_names = {0: 'z', 1: 'y', 2: 'x'}

    for zarr_path in zarr_paths:
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')

        try:
            zarr_root = zarr.open(zarr_path, mode='r')
        except Exception as e:
            print(f"Skipping {zarr_path}: {e}")
            continue

        # 1. Find the earliest reconstruction
        recon_keys = [k for k in zarr_root.keys() if k.startswith('recon-')]
        if not recon_keys: continue
        
        earliest_recon = sorted(recon_keys, key=get_recon_sort_key)[0]
        em_path = f"{earliest_recon}/em"
        if em_path not in zarr_root: continue

        # 2. Find the best EM subfolder based on priority
        em_subfolders = list(zarr_root[em_path].keys())
        if not em_subfolders: continue

        best_em_subfolder = sorted(em_subfolders, key=get_em_subfolder_sort_key)[0]

        # 3. Target only the highest resolution (s0)
        s0_path = f"{em_path}/{best_em_subfolder}/s0"
        if s0_path not in zarr_root: continue

        volume_identifier = f"{dataset_name}/{earliest_recon}/{best_em_subfolder}"
        
        print(f"Loading {volume_identifier} entirely into RAM...")
        
        # Load the entire volume into memory at once
        active_array = zarr_root[s0_path][:] 
        print(f"Loaded shape {active_array.shape} successfully. Generating slices...")

        # 4. Yield 2D slices along Z, Y, and X
        for axis in [0, 1, 2]:
            num_slices = active_array.shape[axis]
            
            for slice_idx in range(num_slices):
                if axis == 0:
                    slice_2d = active_array[slice_idx, :, :]
                elif axis == 1:
                    slice_2d = active_array[:, slice_idx, :]
                else:
                    slice_2d = active_array[:, :, slice_idx]

                yield {
                    "image": Image.fromarray(slice_2d),
                    "crop_name": volume_identifier,
                    "axis": axis_names[axis],
                    "slice": slice_idx
                }

        # Clean up memory explicitly before loading the next volume
        del active_array

def main():
    root_directory = "/lustre/blizzard/stf218/scratch/emin/seg3d/data"
    repo_id = "eminorhan/openorganelle-2d"

    features = Features({
        "image": HFImage(),
        "crop_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32")
    })

    print("Initializing dataset generation. Brute-force RAM loading enabled...")
    
    dataset = Dataset.from_generator(
        generate_2d_slices, 
        gen_kwargs={"root_dir": root_directory}, 
        features=features
    )
    
    print(f"Dataset generated with {len(dataset)} slices.")
    
    dataset = dataset.shuffle(seed=42)
    print("Dataset shuffled!")

    dataset.push_to_hub(repo_id, max_shard_size="1GB")
    print("Dataset uploaded to HF Hub!")

if __name__ == "__main__":
    main()