import os
import zarr
import numpy as np
import scipy.ndimage

from typing import List
from glob import glob

import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2


def make_transform():
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return v2.Compose([to_float, normalize])

def make_transform_3d():
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=(0.449,), std=(0.226,))
    return v2.Compose([to_float, normalize])

transform = make_transform()
transform_3d = make_transform_3d()

class ZarrSegmentationDataset(IterableDataset):
    """
    A simple dataloader for 3D EM segmentation datasets stored in Zarr format.

    This dataset class scans a root directory to find pairs of raw EM volumes
    and their corresponding labeled segmentation crops. It returns fixed-size
    crops suitable for training deep learning models.
    """
    def __init__(self, root_dir, crop_size, rank, base_seed, raw_scale='s0', labels_scale='s0'):
        """
        Initializes the dataset by scanning for valid data samples.

        Args:
            root_dir (str): The path to the root directory containing the Zarr datasets.
            crop_size (tuple, optional): The desired (Z, Y, X) output size of the raw and label crops.
            rank (int): Rank of this process.
            base_seed (int): A base seed for reproducibility.
            raw_scale (str, optional): Resolution for raw data. Defaults to 's0' (highest resolution).
            labels_scale (str, optional): Resolution for labels. Defaults to 's0' (highest resolution).
        """
        super().__init__()        
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.rank = rank
        self.raw_scale = raw_scale
        self.labels_scale = labels_scale
        self.samples = self._find_samples()

        # set rng for this rank (base_seed will change from run to run)
        self.rng = np.random.default_rng(base_seed + self.rank)

        if not self.samples:
            print(f"Warning: No valid samples found in {self.root_dir}. Please check the directory structure and file paths.")

    def _find_samples(self):
        """
        Scans the root directory to find all (raw_volume_group, label_crop) pairs.

        Returns:
            list: A list of dictionaries, where each dictionary contains paths and metadata for a single sample.
        """
        samples = []
        
        # Find all top-level zarr directories
        zarr_paths = glob(os.path.join(self.root_dir, '*/*.zarr'))

        for zarr_path in zarr_paths:
            try:
                zarr_root = zarr.open(zarr_path, mode='r')
            except Exception as e:
                print(f"Could not open {zarr_path}, skipping. Error: {e}")
                continue

            # Iterate through reconstruction groups (e.g., recon-1)
            for recon_name in zarr_root.keys():
                if not recon_name.startswith('recon-'):
                    continue

                raw_group_path_str = os.path.join(recon_name, 'em', 'fibsem-uint8')
                labels_base_path_str = os.path.join(recon_name, 'labels', 'groundtruth')

                if raw_group_path_str not in zarr_root or labels_base_path_str not in zarr_root:
                    continue
                
                # Find all available crops for this reconstruction
                groundtruth_group = zarr_root[labels_base_path_str]
                for crop_name in groundtruth_group.keys():
                    if not crop_name.startswith('crop'):
                        continue
                    
                    # We will use the 'all' mask which contains all label classes
                    label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', self.labels_scale)
                    
                    if label_path_str in zarr_root:
                        samples.append({'zarr_path': zarr_path, 'raw_path_group': raw_group_path_str, 'label_path': label_path_str})

        return samples
    
    def _parse_ome_ngff_metadata(self, attrs, scale_level_name):
        """Helper function to parse scale and translation from OME-NGFF metadata."""
        try:
            multiscales = attrs['multiscales'][0]
            datasets = multiscales['datasets']
            scale_metadata = next((d for d in datasets if d['path'] == scale_level_name), None)
            
            if scale_metadata:
                transformations = scale_metadata['coordinateTransformations']
                scale_transform = next((t for t in transformations if t['type'] == 'scale'), None)
                translation_transform = next((t for t in transformations if t['type'] == 'translation'), None)
                
                scale = scale_transform['scale'] if scale_transform else [1.0, 1.0, 1.0]
                translation = translation_transform['translation'] if translation_transform else [0.0, 0.0, 0.0]
                
                return scale, translation
        except (KeyError, IndexError, StopIteration):
            pass # We will handle the error outside this function
        
        return None, None
    
    def _find_best_raw_scale(self, target_label_scale, raw_attrs):
        """
        Finds the best raw scale level to use based on the target label scale.

        It prioritizes raw scales that are higher or equal resolution (smaller or equal scale value)
        than the target, picking the one with the closest resolution to minimize downsampling.
        """
        try:
            multiscales = raw_attrs['multiscales'][0]
            datasets = multiscales['datasets']
        except (KeyError, IndexError):
            return self.raw_scale, None, None

        available_scales = []
        for d in datasets:
            try:
                scale = next(t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')
                translation = next(t['translation'] for t in d['coordinateTransformations'] if t['type'] == 'translation')
                available_scales.append({'path': d['path'], 'scale': scale, 'translation': translation})
            except (KeyError, StopIteration):
                continue
        
        if not available_scales:
            return self.raw_scale, None, None

        # Find candidate scales where raw_resolution >= label_resolution (raw_scale <= label_scale)
        candidates = [s for s in available_scales if all(rs <= ls for rs, ls in zip(s['scale'], target_label_scale))]

        if candidates:
            # From the candidates, find the one closest to the target scale (minimizing the difference)
            best_candidate = min(candidates, key=lambda s: sum(ls - rs for ls, rs in zip(target_label_scale, s['scale'])))
            return best_candidate['path'], best_candidate['scale'], best_candidate['translation']
        else:
            # No suitable candidate for downsampling, so we'll have to upsample.
            # Pick the highest resolution available (smallest scale values).
            highest_res_scale = min(available_scales, key=lambda s: sum(s['scale']))
            return highest_res_scale['path'], highest_res_scale['scale'], highest_res_scale['translation']

    def _get_sample(self):
        """
        Fetches a single raw crop and its corresponding segmentation mask, both at a fixed output size.
        """
        sample_info = sample_info = self.rng.choice(self.samples)
       
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        label_array = zarr_root[sample_info['label_path']]

        # Parse label metadata
        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
        if label_scale is None:
             raise ValueError(f"Could not parse required OME-NGFF metadata from {label_attrs_group_path}")
            
        # Dynamically find the best raw scale based on the ORIGINAL label scale
        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        best_raw_scale_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)

        if raw_scale is None: # Fallback if metadata parsing failed in helper
             _, raw_scale, raw_translation = self._parse_ome_ngff_metadata(raw_attrs, best_raw_scale_path)
             if raw_scale is None:
                 print(f"Warning: Could not parse metadata for raw volume. Assuming scale=[1,1,1] and translation=[0,0,0].")
                 raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        
        original_shape = label_array.shape
        target_shape = self.crop_size

        # ====== Adjust the label mask to the target output size ======
        # Case 1: The original label mask is larger than the target size, so we take a random crop.
        if all(os >= ts for os, ts in zip(original_shape, target_shape)):
            start_z = self.rng.integers(0, original_shape[0] - target_shape[0] + 1)
            start_y = self.rng.integers(0, original_shape[1] - target_shape[1] + 1)
            start_x = self.rng.integers(0, original_shape[2] - target_shape[2] + 1)
            start_voxels_label = (start_z, start_y, start_x)

            slicing = tuple(slice(start, start + size) for start, size in zip(start_voxels_label, target_shape))
            final_label_mask = label_array[slicing]
            
            offset_physical = [start * scale for start, scale in zip(start_voxels_label, label_scale)]
            adjusted_label_translation = [orig + off for orig, off in zip(label_translation, offset_physical)]
            adjusted_label_scale = label_scale
        
        # Case 2: The label mask is smaller (or mixed), so we must resample it.
        else:
            label_data = label_array[:]
            zoom_factor = [t / s for t, s in zip(target_shape, original_shape)]

            # Use order=0 for nearest-neighbor interpolation to preserve integer labels
            resampled_label_mask = scipy.ndimage.zoom(label_data, zoom_factor, order=0, prefilter=False)
            
            final_label_mask = np.zeros(target_shape, dtype=resampled_label_mask.dtype)
            slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, resampled_label_mask.shape))
            final_label_mask[slicing_for_copy] = resampled_label_mask[slicing_for_copy]

            adjusted_label_translation = label_translation
            original_physical_size = [sh * sc for sh, sc in zip(original_shape, label_scale)]
            adjusted_label_scale = [ps / ts for ps, ts in zip(original_physical_size, target_shape)]
            
        # Now fetch the corresponding raw data using the optimal raw scale
        best_raw_array_path = os.path.join(raw_group_path, best_raw_scale_path)
        raw_array = zarr_root[best_raw_array_path]

        scale_ratio = [ls / rs for ls, rs in zip(adjusted_label_scale, raw_scale)]
        relative_start_physical = [lt - rt for lt, rt in zip(adjusted_label_translation, raw_translation)]
        start_voxels_raw = [int(round(p / s)) for p, s in zip(relative_start_physical, raw_scale)]

        is_downsampling_or_equal = all(r >= 0.999 for r in scale_ratio)

        if is_downsampling_or_equal:
            step = [int(round(r)) for r in scale_ratio]
            step = [max(1, s) for s in step]
            end_voxels_raw = [st + (dim * sp) for st, dim, sp in zip(start_voxels_raw, target_shape, step)]
            slicing = tuple(slice(st, en, sp) for st, en, sp in zip(start_voxels_raw, end_voxels_raw, step))
            raw_crop_from_zarr = raw_array[slicing]
        else:
            label_physical_size = [sh * sc for sh, sc in zip(target_shape, adjusted_label_scale)]
            relative_end_physical = [s + size for s, size in zip(relative_start_physical, label_physical_size)]
            end_voxels_raw = [int(round(p / s)) for p, s in zip(relative_end_physical, raw_scale)]
            slicing = tuple(slice(start, end) for start, end in zip(start_voxels_raw, end_voxels_raw))
            raw_crop = raw_array[slicing]

            if any(s == 0 for s in raw_crop.shape):
                raw_crop_from_zarr = np.zeros(target_shape, dtype=raw_array.dtype)
            else:
                zoom_factor = [t / s for t, s in zip(target_shape, raw_crop.shape)]
                raw_crop_from_zarr = scipy.ndimage.zoom(raw_crop, zoom_factor, order=1, prefilter=False)

        final_raw_crop = np.zeros(target_shape, dtype=raw_array.dtype)
        slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, raw_crop_from_zarr.shape))
        final_raw_crop[slicing_for_copy] = raw_crop_from_zarr[slicing_for_copy]

        # Add channel axis (TODO: need to add input/label transformations here)
        raw_tensor = torch.from_numpy(final_raw_crop[np.newaxis, ...]).float() / 255.0
        label_tensor = torch.from_numpy(final_label_mask).long()

        return transform_3d(raw_tensor), label_tensor

    def __iter__(self):
        while True:
            yield self._get_sample()


class ZarrSegmentationDataset2D(ZarrSegmentationDataset):
    """
    An iterable dataloader that provides random 2D slices from a collection of 3D Zarr volumes.
    It inherits seeding and iteration logic from its 3D parent class.
    """
    def __init__(self, root_dir, crop_size, rank, base_seed, raw_scale='s0', labels_scale='s0'):
        """
        Initializes the 2D dataloader.

        Args:
            root_dir (str): Path to the root directory containing Zarr datasets.
            crop_size (tuple): Desired (H, W) output size for 2D slices.
            rank (int): The distributed rank of the current process.
            base_seed (int): A base seed for reproducibility.
            raw_scale (str, optional): Default highest-resolution scale for raw data.
            labels_scale (str, optional): Scale level for labels.
        """
        # Call the parent constructor, but we will use our own 2D crop_size.
        # The parent's crop_size will be ignored since we override _get_single_item.
        super().__init__(root_dir, None, rank, base_seed, raw_scale='s0', labels_scale='s0')
        self.crop_size = crop_size # This is a 2D tuple (H, W)

    def _get_sample(self):
        """
        Fetches a single random 2D slice, loading it directly from the Zarr store.
        This method overrides the parent class's method.
        """
        # Randomly select a 3D volume, axis, and slice index
        sample_info = self.rng.choice(self.samples)
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        label_array_3d = zarr_root[sample_info['label_path']]
        shape_3d = label_array_3d.shape
        
        axis = self.rng.integers(0, 3) # Randomly choose Z, Y, or X axis
        slice_idx = self.rng.integers(0, shape_3d[axis]) # Randomly choose slice along that axis

        # Get the 2D label slice and its 3D metadata
        slicing_3d = [slice(None)] * 3
        slicing_3d[axis] = slice_idx
        label_slice_2d = label_array_3d[tuple(slicing_3d)]

        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale_3d, label_translation_3d = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)

        # Adjust label slice to crop_size (crop or resample)
        original_shape_2d = label_slice_2d.shape
        target_shape_2d = self.crop_size
        if all(os >= ts for os, ts in zip(original_shape_2d, target_shape_2d)):
            start_h = self.rng.integers(0, original_shape_2d[0] - target_shape_2d[0] + 1)
            start_w = self.rng.integers(0, original_shape_2d[1] - target_shape_2d[1] + 1)
            final_label_slice = label_slice_2d[start_h:start_h+target_shape_2d[0], start_w:start_w+target_shape_2d[1]]
            
            axes_2d = [i for i in range(3) if i != axis]
            offset_physical = [start_h * label_scale_3d[axes_2d[0]], start_w * label_scale_3d[axes_2d[1]]]
            adjusted_label_translation_2d = [label_translation_3d[axes_2d[0]] + offset_physical[0], label_translation_3d[axes_2d[1]] + offset_physical[1]]
            adjusted_label_scale_2d = [label_scale_3d[axes_2d[0]], label_scale_3d[axes_2d[1]]]
        else:
            zoom_factor = [t / s for t, s in zip(target_shape_2d, original_shape_2d)]
            final_label_slice = scipy.ndimage.zoom(label_slice_2d, zoom_factor, order=0, prefilter=False)

            axes_2d = [i for i in range(3) if i != axis]
            adjusted_label_translation_2d = [label_translation_3d[axes_2d[0]], label_translation_3d[axes_2d[1]]]
            original_physical_size_2d = [sh * sc for sh, sc in zip(original_shape_2d, [label_scale_3d[d] for d in axes_2d])]
            adjusted_label_scale_2d = [ps / ts for ps, ts in zip(original_physical_size_2d, target_shape_2d)]

        # Find best raw scale and fetch corresponding 2D raw slice
        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        
        temp_target_label_scale_3d = [0,0,0]
        axes_2d = [i for i in range(3) if i != axis]
        temp_target_label_scale_3d[axes_2d[0]] = adjusted_label_scale_2d[0]
        temp_target_label_scale_3d[axes_2d[1]] = adjusted_label_scale_2d[1]
        temp_target_label_scale_3d[axis] = label_scale_3d[axis]

        best_raw_scale_path, raw_scale_3d, raw_translation_3d = self._find_best_raw_scale(temp_target_label_scale_3d, raw_attrs)
        raw_array_3d = zarr_root[os.path.join(raw_group_path, best_raw_scale_path)]
        
        phys_start_3d = [0,0,0]
        phys_start_3d[axes_2d[0]] = adjusted_label_translation_2d[0]
        phys_start_3d[axes_2d[1]] = adjusted_label_translation_2d[1]
        phys_start_3d[axis] = label_translation_3d[axis] + slice_idx * label_scale_3d[axis]
        
        relative_phys_start_3d = [ps - rt for ps, rt in zip(phys_start_3d, raw_translation_3d)]
        start_voxels_raw_3d = [int(round(p / s)) for p, s in zip(relative_phys_start_3d, raw_scale_3d)]

        size_in_phys_2d = [sh * sc for sh, sc in zip(target_shape_2d, adjusted_label_scale_2d)]
        size_in_raw_voxels_2d = [int(round(p / s)) for p, s in zip(size_in_phys_2d, [raw_scale_3d[d] for d in axes_2d])]
        
        # Get the actual shape of the raw data array
        raw_shape_3d = raw_array_3d.shape

        # Clamp the calculated coordinates to be within the valid bounds of the raw array (this prevents the BoundsCheckError)s
        safe_start_voxels_raw_3d = [np.clip(start_voxels_raw_3d[i], 0, raw_shape_3d[i] - 1) for i in range(3)]
        
        safe_raw_slicing = [0, 0, 0]
        safe_raw_slicing[axes_2d[0]] = slice(safe_start_voxels_raw_3d[axes_2d[0]], min(safe_start_voxels_raw_3d[axes_2d[0]] + size_in_raw_voxels_2d[0], raw_shape_3d[axes_2d[0]]))
        safe_raw_slicing[axes_2d[1]] = slice(safe_start_voxels_raw_3d[axes_2d[1]], min(safe_start_voxels_raw_3d[axes_2d[1]] + size_in_raw_voxels_2d[1], raw_shape_3d[axes_2d[1]]))
        safe_raw_slicing[axis] = safe_start_voxels_raw_3d[axis]

        # Use the safe, clamped slicing to read from Zarr
        raw_slice_2d = raw_array_3d[tuple(safe_raw_slicing)]
        if raw_slice_2d.shape != target_shape_2d:
             if any(s == 0 for s in raw_slice_2d.shape):
                 final_raw_slice = np.zeros(target_shape_2d, dtype=raw_array_3d.dtype)
             else:
                zoom_factor = [t / s for t, s in zip(target_shape_2d, raw_slice_2d.shape)]
                final_raw_slice = scipy.ndimage.zoom(raw_slice_2d, zoom_factor, order=1, prefilter=False)
        else:
            final_raw_slice = raw_slice_2d

        # Add channel axis (TODO: need to add input/label transformations here)
        raw_tensor = torch.from_numpy(final_raw_slice[np.newaxis, ...]).float() / 255.0
        label_tensor = torch.from_numpy(final_label_slice).long()
        
        return transform(raw_tensor.expand(3, -1, -1)), label_tensor


def build_data_loader(
    batch_size: int,
    root_dir: str,
    crop_size: tuple[int, int] | tuple[int, int, int], 
    rank: int = 0,
    base_seed: int = 0
) -> DataLoader:
    """
    Builds a volume EM data loader for distributed training.

    Args:
        batch_size (int): The batch size for the data loader.
        crop_size: A tuple of (depth, height, width) for the desired crop.
        rank (int): The rank of the current process.

    Returns:
        DataLoader: A configured data loader for distributed training.
    """
    # We pick out 2D or 3D dataset class based on the crop_size argument
    if len(crop_size) == 3:
        dataset = ZarrSegmentationDataset(
            root_dir=root_dir, 
            crop_size=crop_size,
            rank=rank,
            base_seed=base_seed
        )
    else:
        dataset = ZarrSegmentationDataset2D(
            root_dir=root_dir, 
            crop_size=crop_size,
            rank=rank,
            base_seed=base_seed
        )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)