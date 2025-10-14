import torch
from dinov3.eval.segmentation.models import build_segmentation_decoder

DINOv3_REPO_DIR = "/lustre/gale/stf218/scratch/emin/dinov3"
backbone_arch = 'dinov3_vit7b16'  # backbone type
head_arch = 'linear'  # decoder head type

backbone = torch.hub.load(DINOv3_REPO_DIR, backbone_arch, source="local", pretrained=False) #weights=None, backbone_weights=None)
segmentor = build_segmentation_decoder(backbone, decoder_type=head_arch)
