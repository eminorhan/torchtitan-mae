import os
import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from dinov3.eval.segmentation.models import build_segmentation_decoder
from pathlib import Path

TORCH_HUB_PATH = "/lustre/blizzard/stf218/scratch/emin/torch_hub"  # this is where the dinov3 pth checkpoints are stored
DINOV3_REPO_PATH = "/lustre/blizzard/stf218/scratch/emin/dinov3"  # dinov3 repo path
DECODER_TYPE = "linear"  # segmentation head type
NUM_CLASSES = 64  # number of classes in semantic segmentation task

BACKBONE_CKPT_DICT = {
    # "dinov3_vit7b16_3D": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    # "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    # "dinov3_vith16plus_3D": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    # "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vitl16_3D": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    # "dinov3_vitb16_3D": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    # "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

BACKBONE_PTH_ROOT = "/lustre/blizzard/stf218/scratch/emin/torch_hub/checkpoints"
DCP_ROOT = "/lustre/blizzard/stf218/scratch/emin/torchtitan-mae/outputs"

torch.hub.set_dir(TORCH_HUB_PATH)

for backbone, backbone_pth in BACKBONE_CKPT_DICT.items():
    bbone = torch.hub.load(DINOV3_REPO_PATH, backbone, source="local", weights=f"{BACKBONE_PTH_ROOT}/{backbone_pth}", pretrained=False, use_fa3=True)
    model = build_segmentation_decoder(bbone, decoder_type=DECODER_TYPE, num_classes=NUM_CLASSES)
    model_state_dict = model.state_dict()
    print(f"Loaded and built model {backbone}...")
    print(f"Model: {model}")

    dcp_path = Path(f"{DCP_ROOT}/{backbone}_{DECODER_TYPE}_scratch/checkpoint/step-0")
    storage_writer = DCP.filesystem.FileSystemWriter(dcp_path, thread_count=1)
    dcp_path.mkdir(parents=True, exist_ok=True)
    DCP.save({"model": model_state_dict}, storage_writer=storage_writer)
    print(f"Wrote DCP ckpt to {dcp_path}...")