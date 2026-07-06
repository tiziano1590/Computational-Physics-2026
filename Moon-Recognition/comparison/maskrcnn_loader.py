"""Rebuild Amir's custom Mask R-CNN and load his checkpoint.

The architecture (custom deep heads) lives only inside amir_rcnn.ipynb; this
module reproduces it so the model can be loaded outside the notebook. Head
definitions mirror cell 23 of that notebook.
"""
import types
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator


class ResidualConvBlock(nn.Module):
    def __init__(s, ch=256, groups=8, dropout=0.05):
        super().__init__()
        s.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch))
        s.act = nn.ReLU(inplace=True)
    def forward(s, x): return s.act(x + s.block(x))


class DeepResidualMaskHead(nn.Module):
    def __init__(s, in_ch=256, hidden=256, num_blocks=4, groups=8, dropout=0.05):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
                               nn.GroupNorm(groups, hidden), nn.ReLU(inplace=True))
        s.blocks = nn.Sequential(*[ResidualConvBlock(hidden, groups, dropout) for _ in range(num_blocks)])
        s.out_channels = hidden
    def forward(s, x): return s.blocks(s.stem(x))


class DeepMaskPredictor(nn.Module):
    def __init__(s, in_ch=256, hidden=256, num_classes=3, groups=8, dropout=0.05):
        super().__init__()
        s.refine = nn.Sequential(ResidualConvBlock(in_ch, groups, dropout),
                                 nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
                                 nn.GroupNorm(groups, hidden), nn.ReLU(inplace=True))
        s.up = nn.ConvTranspose2d(hidden, hidden, 2, stride=2)
        s.act = nn.ReLU(inplace=True); s.drop = nn.Dropout2d(dropout)
        s.mask_logits = nn.Conv2d(hidden, num_classes, 1)
    def forward(s, x): return s.mask_logits(s.drop(s.act(s.up(s.refine(x)))))


class DeepBoxHead(nn.Module):
    def __init__(s, in_ch=256, roi=7, hidden=1024, depth=4, dropout=0.10):
        super().__init__()
        layers = [nn.Flatten()]; cur = in_ch * roi * roi
        for _ in range(depth):
            layers += [nn.Linear(cur, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]; cur = hidden
        s.layers = nn.Sequential(*layers); s.out_channels = hidden
    def forward(s, x): return s.layers(x)


def build_and_load(weights_path, device="cpu", tile_size=256):
    """Returns (model.eval() on device, class_names, cfg). Asserts a clean load."""
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = types.SimpleNamespace(**ck["config"]); names = ck["class_names"]; ncls = len(names)
    m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                              trainable_backbone_layers=cfg.trainable_backbone_layers)
    out_ch = m.backbone.out_channels
    roi = m.roi_heads.box_roi_pool.output_size; roi = int(roi[0]) if isinstance(roi, (tuple, list)) else int(roi)
    m.roi_heads.box_head = DeepBoxHead(out_ch, roi, cfg.box_head_hidden, cfg.box_head_depth, cfg.box_head_dropout)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_head.out_channels, ncls)
    m.roi_heads.mask_head = DeepResidualMaskHead(out_ch, cfg.mask_head_hidden, cfg.mask_head_blocks, dropout=cfg.head_dropout)
    m.roi_heads.mask_predictor = DeepMaskPredictor(m.roi_heads.mask_head.out_channels, cfg.mask_head_hidden, ncls, dropout=cfg.head_dropout)
    sizes = tuple((s,) for s in cfg.anchor_sizes)
    m.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(tuple(cfg.anchor_ratios),) * len(sizes))
    m.transform.min_size = (tile_size,); m.transform.max_size = tile_size * 2
    m.roi_heads.score_thresh = 0.0   # keep all detections; we threshold by score ourselves
    m.roi_heads.nms_thresh = cfg.box_nms_thresh
    m.roi_heads.detections_per_img = cfg.detections_per_img
    res = m.load_state_dict(ck["model_state"], strict=False)
    assert not res.missing_keys, f"Mask R-CNN load mismatch: {len(res.missing_keys)} missing"
    return m.to(device).eval(), names, cfg
