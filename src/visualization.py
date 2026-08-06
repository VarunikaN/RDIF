"""Utilities for exporting RDIF-CAM visualizations."""

from pathlib import Path

import torch
from PIL import Image


def _prepare_image(image):
    image = image.detach().cpu()[0].float()
    image = (image - image.amin()) / (image.amax() - image.amin()).clamp_min(1e-8)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image[:3].permute(1, 2, 0)


def _heatmap(cam):
    cam = cam.detach().cpu()[0, 0].float().clamp(0, 1)
    red = (1.5 - (4 * cam - 3).abs()).clamp(0, 1)
    green = (1.5 - (4 * cam - 2).abs()).clamp(0, 1)
    blue = (1.5 - (4 * cam - 1).abs()).clamp(0, 1)
    return torch.stack((red, green, blue), dim=-1)


def _save(image, path):
    Image.fromarray((image.clamp(0, 1) * 255).to(torch.uint8).numpy()).save(path)


def save_visualizations(image, cam, output_dir, prefix="", alpha=0.45):
    """Save the input, heatmap, overlay, and a paper-ready comparison panel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = _prepare_image(image)
    heatmap = _heatmap(cam)
    overlay = (1 - alpha) * original + alpha * heatmap

    filename = lambda name: f"{prefix}_{name}.png" if prefix else f"{name}.png"
    _save(heatmap, output_dir / filename("heatmap"))
    _save(overlay, output_dir / filename("overlay"))
    _save(torch.cat((original, heatmap, overlay), dim=1), output_dir / filename("comparison"))


def save_overlay(image, cam, output_path, alpha=0.45):
    """Save only an RDIF-CAM overlay for lightweight integrations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = _prepare_image(image)
    overlay = (1 - alpha) * original + alpha * _heatmap(cam)
    _save(overlay, output_path)
