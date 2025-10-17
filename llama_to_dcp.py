import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from dinov3.eval.segmentation.models import build_segmentation_decoder

TORCH_HUB_PATH = "/lustre/gale/stf218/scratch/emin/torch_hub"
DINOV3_REPO_PATH = "/lustre/gale/stf218/scratch/emin/dinov3"
TORCH_SAVE_PATH = "/lustre/gale/stf218/scratch/emin/torch_hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
BACKBONE_ARCH = "dinov3_vit7b16"
DECODER_TYPE = "linear"
DCP_CKPT_PATH = "/lustre/gale/stf218/scratch/emin/torchtitan-mae/outputs/demo/checkpoint/step-0"

torch.hub.set_dir(TORCH_HUB_PATH)
backbone = torch.hub.load(DINOV3_REPO_PATH, BACKBONE_ARCH, source="local", weights=TORCH_SAVE_PATH)
model = build_segmentation_decoder(backbone, decoder_type=DECODER_TYPE, num_classes=99)
model_state_dict = model.state_dict()
print(f"Loaded and built model...")

storage_writer = DCP.filesystem.FileSystemWriter(DCP_CKPT_PATH, thread_count=1)
DCP.save({"model": model_state_dict}, storage_writer=storage_writer)
print(f"Wrote DCP ckpt to {DCP_CKPT_PATH}")

# torch_save_to_dcp(TORCH_SAVE_PATH, DCP_CKPT_PATH)