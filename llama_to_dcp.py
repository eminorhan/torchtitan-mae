from torch.distributed.checkpoint.format_utils import torch_save_to_dcp

TORCH_SAVE_PATH = "/lustre/gale/stf218/scratch/emin/torch_hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
DCP_CKPT_PATH = "/lustre/gale/stf218/scratch/emin/torchtitan-mae/outputs/demo/checkpoint/step-0"

torch_save_to_dcp(TORCH_SAVE_PATH, DCP_CKPT_PATH)