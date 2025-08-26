import os, time, uuid, json
import random
import pickle
import zarr
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from torchtitan.logging import logger
from multiprocessing.shared_memory import SharedMemory

BASE_DIR = "/lustre/gale/stf218/scratch/emin/seg3d/data"
_volumes_list = [
    os.path.join(BASE_DIR, "jrc_mus-granule-neurons-1/jrc_mus-granule-neurons-1.zarr/recon-2/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-granule-neurons-3/jrc_mus-granule-neurons-3.zarr/recon-2/em/fibsem-int16/s0"),
    os.path.join(BASE_DIR, "jrc_mus-hippocampus-2/jrc_mus-hippocampus-2.zarr/recon-1/em/fibsem-uint8/s0"),
    os.path.join(BASE_DIR, "jrc_mus-hippocampus-3/jrc_mus-hippocampus-3.zarr/recon-1/em/fibsem-int16/s0"),
    ]

# _volumes_list = [
#     os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0"),
#     os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0"),
#     os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0"),
#     os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0"),
#     os.path.join(BASE_DIR, "jrc_mus-granule-neurons-2/jrc_mus-granule-neurons-2.zarr/recon-2/em/fibsem-int16/s0")
#     ]

class VolumeDataset(IterableDataset):
    """PyTorch IterableDataset for generating random crops from volume data on-the-fly.

    This dataset generates random samples from EM volumes.

    Args:
        crop_size: A tuple of (depth, height, width) for the desired crops.
        world_size (int): number of data parallel processes participating in distributed training
        rank (int): rank of the current data parallel process
    """
    def __init__(self, crop_size, rank: int, world_size: int):
        self.crop_size = crop_size
        self.rank = rank
        self.world_size = world_size
        self.volume = None
        self.shm = None

        # ---------- topology ----------
        num_gpus_per_node = 4  # set correctly
        node_rank = rank // num_gpus_per_node
        local_rank = rank % num_gpus_per_node
        is_local_leader = (local_rank == 0)

        meta_path = f"/dev/shm/vol_node{node_rank}.meta"   # per-node metadata file
        tmp_meta_path = meta_path + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"

        # dtype maps (symmetric)
        dtype_map = {str(np.dtype("uint8")): 1, str(np.dtype("int16")): 2, str(np.dtype("float32")): 3}
        dtype_map_rev = {1: np.uint8, 2: np.int16, 3: np.float32}
        payload = None

        if is_local_leader:
            # 1) pick and load assigned volume
            indices = np.arange(len(_volumes_list))
            rng = np.random.RandomState(42)
            rng.shuffle(indices)
            assigned_volume_index = indices[node_rank % len(indices)]
            volume_path = _volumes_list[assigned_volume_index]

            print(f"[leader node{node_rank} global{rank}] loading {volume_path}", flush=True)
            zarr_volume = zarr.open(volume_path, mode="r")
            temp_volume = np.array(zarr_volume, copy=True)

            # 2) create unique SHM name and create shared memory
            unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            shm_name = f"vol_node{node_rank}_{unique_suffix}"
            shm_size = temp_volume.nbytes

            print(f"[leader node{node_rank} global{rank}] creating SHM '{shm_name}' size={shm_size}", flush=True)
            self.shm = SharedMemory(create=True, name=shm_name, size=shm_size)
            shared_volume_np = np.ndarray(temp_volume.shape, dtype=temp_volume.dtype, buffer=self.shm.buf)
            shared_volume_np[...] = temp_volume
            self.volume = shared_volume_np

            # 3) prepare payload and atomically write JSON metadata to /dev/shm
            dtype_code = dtype_map[str(np.dtype(self.volume.dtype))]
            payload = {
                "shm_name": shm_name,
                "shape": tuple(int(x) for x in self.volume.shape),
                "dtype_code": int(dtype_code),
            }
            print(f"[leader node{node_rank} global{rank}] payload={payload}", flush=True)

            # atomic write: write to tmp file then rename
            with open(tmp_meta_path, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_meta_path, meta_path)   # atomic on POSIX
            print(f"[leader node{node_rank} global{rank}] wrote meta to {meta_path}", flush=True)

        else:
            # non-leaders: wait for the file to appear (poll)
            MAX_WAIT = 3600.0   # seconds
            POLL = 0.05
            waited = 0.0
            while not os.path.exists(meta_path) and waited < MAX_WAIT:
                time.sleep(POLL)
                waited += POLL
            if not os.path.exists(meta_path):
                raise RuntimeError(f"[node{node_rank} global{rank}] timed out waiting for metadata file {meta_path}")

            # read the payload
            with open(meta_path, "r") as f:
                payload = json.load(f)
            print(f"[node{node_rank} global{rank}] read payload={payload}", flush=True)

        # Now attach based on payload (both leader and non-leaders will have payload)
        shm_name = payload["shm_name"]
        shape = tuple(int(x) for x in payload["shape"])
        dtype_code = int(payload["dtype_code"])
        dtype = dtype_map_rev[dtype_code]

        if not is_local_leader:
            # try attach (with some retries because leader might still be finishing)
            MAX_TRIES = 20
            ATTEMPT_DELAY = 0.05
            attached = False
            last_exc = None
            for i in range(MAX_TRIES):
                try:
                    self.shm = SharedMemory(name=shm_name, create=False)
                    self.volume = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
                    attached = True
                    print(f"[node{node_rank} global{rank}] attached to SHM {shm_name} on attempt {i+1}", flush=True)
                    break
                except FileNotFoundError as e:
                    last_exc = e
                    time.sleep(ATTEMPT_DELAY)
                except Exception as e:
                    last_exc = e
                    time.sleep(ATTEMPT_DELAY)

            if not attached:
                raise RuntimeError(f"[node{node_rank} global{rank}] failed to attach to SharedMemory '{shm_name}': {last_exc}")

        else:
            # leader already has self.volume set from the write above
            print(f"[leader node{node_rank} global{rank}] leader keeps SHM {shm_name}", flush=True)

        # Optional: if you want to clean up the metadata file later, let leader remove it
        # but don't unlink SHM until you're certain all children finished (e.g., at shutdown)
        # Final global barrier to ensure everyone is ready (optional)
        print(f"[node{node_rank} global{rank}] setup complete. volume shape: {self.volume.shape}", flush=True)


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
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)