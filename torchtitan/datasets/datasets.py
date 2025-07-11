import os
import random
import pickle
import zarr
import numpy as np
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.logging import logger


class VolumeDataset(IterableDataset, Stateful):
    """PyTorch IterableDataset for generating random crops from volume data on-the-fly.

    This dataset generates random matrices, simulating a stream of data for training.
    It is stateful and supports checkpointing to ensure reproducibility in a
    distributed environment.

    Args:
        data_dir: The path to the top-level directory containing volume folders.
        subdir_name: Subdirectory name containing the EM data.
        crop_size: A tuple of (depth, height, width) for the desired crop.
        resolution: Resolution at which to retrieve the data (default: 's0').
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
    """
    def __init__(
        self,
        data_dir: str,
        subdir_name: str,
        crop_size: tuple[int, int, int], 
        resolution: str,
        world_size: int,
        rank: int,
    ) -> None:
        self.data_dir = data_dir
        self.subdir_name = subdir_name
        self.crop_size = crop_size
        self.resolution = resolution
        self.world_size = world_size
        self.rank = rank

        # list of all subdirectories in the data directory
        self.volumes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        # seed the rng for this process to ensure different data per rank
        # adding rank to a random seed ensures that each process starts with a
        # unique, non-overlapping sequence of random numbers
        np.random.seed(rank + np.random.randint(0, 2**32 - 1))

        # variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._rng_state = np.random.get_state()

    def _generate_sample(self) -> np.ndarray:
        """
        Selects a random volume from the data directory, opens its Zarr array,
        and extracts a random 3D crop.

        The expected directory structure for each volume is:
        <data_dir>/<volume_name>/<volume_name>.zarr/recon-1/em/fibsem-uint8/s0

        Returns:
            A NumPy array containing the cropped data, or None if an error occurs.
        """         
        # randomly select one of the volumes
        selected_volume = random.choice(self.volumes)

        # construct the path to the zarr array
        zarr_path = os.path.join(
            self.data_dir,
            selected_volume,
            f"{selected_volume}.zarr",
            self.subdir_name
        )
        
        # open the zarr array at given resolution
        zarr_group = zarr.open(zarr_path, mode='r')                
        zarr_array = zarr_group[self.resolution]
        full_shape = zarr_array.shape
        
        assert all(c <= f for c, f in zip(self.crop_size, full_shape)), "Crop size must be smaller than the full volume along each dimension."

        # calculate the maximum possible starting index for the crop in each dimension
        cz, cy, cx = self.crop_size
        max_z = full_shape[0] - cz
        max_y = full_shape[1] - cy
        max_x = full_shape[2] - cx
        
        # generate a random starting point
        start_z = random.randint(0, max_z)
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        
        # read the specific crop from the zarr array into a NumPy array
        crop_slice = (
            slice(start_z, start_z + cz),
            slice(start_y, start_y + cy),
            slice(start_x, start_x + cx)
        )
        crop = zarr_array[crop_slice] / 255
        crop = torch.from_numpy(crop).unsqueeze(0).to(torch.bfloat16)
        # print(f"Crop max/min/shape/dtype: {crop.max()}/{crop.min()}/{crop.shape}/{crop.dtype}")
        
        return crop

    def __iter__(self):
        # restore the RNG state at the beginning of iteration to ensure
        # that resuming from a checkpoint continues the same random sequence.
        np.random.set_state(self._rng_state)  # TODO: not sure if I really need this

        while True:
            sample = self._generate_sample()
            self._sample_idx += 1

            # yield sample
            yield sample

    def state_dict(self) -> Dict[str, Any]:
        # capture the current RNG state for checkpointing.
        self._rng_state = np.random.get_state()
        return {
            "sample_idx": self._sample_idx,
            "rng_state": self._rng_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._sample_idx = state_dict["sample_idx"]
        self._rng_state = state_dict["rng_state"]
        # rng state will be restored at the start of the next __iter__ call.


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """
    def __init__(self, dp_rank: int, dataset: IterableDataset, batch_size: int):
        super().__init__(dataset, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # state being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}")
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_data_loader(
    data_dir: str,
    subdir_name: str,
    batch_size: int,
    crop_size: tuple[int, int, int], 
    resolution: str = "s0",
    world_size: int = 1,
    rank: int = 0,
) -> DPAwareDataLoader:
    """
    Builds a data loader for distributed training.

    This function can create a data loader for a Hugging Face dataset or a
    dataset with synthetically generated data.

    Args:
        data_dir: The path to the top-level directory containing volume folders.
        subdir_name: Subdirectory name containing the EM data.
        batch_size (int): The batch size for the data loader.
        crop_size: A tuple of (depth, height, width) for the desired crop.
        resolution: Resolution at which to retrieve the data (default: 's0').
        world_size (int): The total number of processes in the distributed group.
        rank (int): The rank of the current process.

    Returns:
        DPAwareDataLoader: A configured stateful data loader for distributed training.
    """
    dataset = VolumeDataset(
        data_dir=data_dir,
        subdir_name=subdir_name,
        crop_size=crop_size, 
        resolution=resolution,
        world_size=world_size,
        rank=rank
    )

    return DPAwareDataLoader(rank, dataset, batch_size=batch_size)