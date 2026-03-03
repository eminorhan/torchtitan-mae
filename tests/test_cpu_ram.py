import os
import psutil
import torch.distributed as dist

# Initialize the distributed process group
# This assumes you've launched your script with torchrun or a similar tool
dist.init_process_group(backend="nccl")

# Get the total number of ranks on the current node
# LOCAL_WORLD_SIZE is typically set by torchrun or other launch scripts
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

# Get total available CPU RAM on this node
mem_info = psutil.virtual_memory()
available_ram_bytes = mem_info.available

# Calculate the RAM available per FSDP rank
ram_per_rank_bytes = available_ram_bytes / local_world_size
ram_per_rank_gb = ram_per_rank_bytes / (1024 ** 3)

print(f"Rank {dist.get_rank()}: Total local ranks = {local_world_size}")
print(f"Rank {dist.get_rank()}: Total available CPU RAM = {mem_info.available / (1024 ** 3):.2f} GB")
print(f"Rank {dist.get_rank()}: Estimated available CPU RAM per FSDP rank = {ram_per_rank_gb:.2f} GB")

# Clean up the process group
dist.destroy_process_group()