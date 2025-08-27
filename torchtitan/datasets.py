import os
import random
import zarr
import numpy as np
from typing import List

import torch
from torch.utils.data import IterableDataset, DataLoader

BASE_DIR = "/lustre/gale/stf218/scratch/emin/seg3d/data"
_volumes_list = [
    os.path.join(BASE_DIR, "jrc_mus-granule-neurons-1/jrc_mus-granule-neurons-1.zarr/recon-2/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-granule-neurons-3/jrc_mus-granule-neurons-3.zarr/recon-2/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-hippocampus-2/jrc_mus-hippocampus-2.zarr/recon-1/em/fibsem-uint8/s0"),
    os.path.join(BASE_DIR, "jrc_mus-hippocampus-3/jrc_mus-hippocampus-3.zarr/recon-1/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-cerebellum-5/jrc_mus-cerebellum-5.zarr/recon-1/em/fibsem-int16/s0"),
    ]

class VolumeDataset(IterableDataset):
    """PyTorch IterableDataset for generating random crops from volume data on-the-fly.

    This dataset generates random samples from EM volumes.

    Args:
        crop_size: A tuple of (depth, height, width) for the desired crops.
        world_size (int): number of data parallel processes participating in distributed training
        rank (int): rank of the current data parallel process
    """
    def __init__(
        self,
        crop_size: tuple[int, int, int], 
        world_size: int,
        rank: int,
    ) -> None:
        self.crop_size = crop_size
        self.world_size = world_size
        self.rank = rank

        # volume path for this rank
        _volume_path = np.random.choice(_volumes_list)
        print(f"[Rank {rank}] selected volume: {_volume_path}")

        # open zarr array for this rank
        self.volume = zarr.open(_volume_path, mode='r')

    def _generate_sample(self) -> torch.Tensor:
        """
        Crops a random subvolume from the loaded volume.

        Returns:
            A torch Tensor containing the cropped data, or None if an error occurs.
        """         
        
        # Get the dimensions of the full volume and the desired crop.
        vol_d, vol_h, vol_w = self.volume.shape
        crop_d, crop_h, crop_w = self.crop_size

        # Ensure the crop size is not larger than the volume itself.
        if any(cs > vs for cs, vs in zip(self.crop_size, self.volume.shape)):
            raise ValueError(f"Crop size {self.crop_size} is larger than the volume shape {self.volume.shape}.")

        # Determine the valid range for the starting coordinates of the crop.
        # The crop must not extend beyond the volume's boundaries.
        max_d = vol_d - crop_d
        max_h = vol_h - crop_h
        max_w = vol_w - crop_w

        # Generate random starting coordinates.
        start_d = np.random.randint(0, max_d + 1)
        start_h = np.random.randint(0, max_h + 1)
        start_w = np.random.randint(0, max_w + 1)

        # Extract the crop using numpy slicing.
        crop = self.volume[
            start_d : start_d + crop_d,
            start_h : start_h + crop_h,
            start_w : start_w + crop_w
        ]

        # crop = zarr_array[crop_slice] / 255 - 0.5  # normalize
        crop = torch.from_numpy(crop).unsqueeze(0).to(torch.bfloat16)
        # print(f"Crop max/min/shape/dtype: {crop.max()}/{crop.min()}/{crop.shape}/{crop.dtype}")
        return crop

    def __iter__(self):
        while True:
            sample = self._generate_sample()
            yield sample

def build_data_loader(
    batch_size: int,
    crop_size: tuple[int, int, int], 
    num_workers: int = 0,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """
    Builds a volume EM data loader for distributed training.

    Args:
        batch_size (int): The batch size for the data loader.
        crop_size: A tuple of (depth, height, width) for the desired crop.
        world_size (int): The total number of processes in the distributed group.
        rank (int): The rank of the current process.

    Returns:
        DataLoader: A configured data loader for distributed training.
    """
    dataset = VolumeDataset(
        crop_size=crop_size,
        world_size=world_size,
        rank=rank,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)