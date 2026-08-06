"""Model-agnostic RDIF-CAM implementation for dense prediction models."""

import math

import torch
import torch.nn.functional as F


def _normalize(value):
    minimum, maximum = value.amin(dim=(-2, -1), keepdim=True), value.amax(dim=(-2, -1), keepdim=True)
    return (value - minimum) / (maximum - minimum).clamp_min(1e-8)


def _gabor_kernel(theta, sigma, device, dtype):
    size = 15
    axis = torch.arange(size, device=device, dtype=dtype) - size // 2
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    return torch.exp(-(x_theta.square() + y_theta.square()) / (2 * sigma**2)) * torch.cos(2 * math.pi * x_theta / 8)


def radiomic_gate(image):
    """Fuse Gabor, LBP, and inverse-local-variance texture features."""
    grayscale = _normalize(image.mean(dim=1, keepdim=True))
    responses = []
    for theta in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        for sigma in (2.0, 4.0):
            kernel = _gabor_kernel(theta, sigma, image.device, image.dtype)[None, None]
            responses.append(F.conv2d(grayscale, kernel, padding=7).abs())
    gabor = _normalize(torch.stack(responses).amax(dim=0))

    lbp = torch.zeros_like(grayscale)
    for angle in range(8):
        dx, dy = round(2 * math.cos(2 * math.pi * angle / 8)), round(-2 * math.sin(2 * math.pi * angle / 8))
        lbp += (grayscale >= torch.roll(grayscale, (dy, dx), dims=(-2, -1))).float()
    lbp = _normalize(lbp / 8)

    mean = F.avg_pool2d(grayscale, 7, stride=1, padding=3)
    variance = F.avg_pool2d((grayscale - mean).square(), 7, stride=1, padding=3)
    homogeneity = _normalize(1 / (1 + variance.sqrt()))
    return torch.sigmoid(6 * ((gabor + lbp + homogeneity) / 3 - 0.5))


def _diffuse(seed, gate, iterations, kappa, step_size):
    result = seed
    for _ in range(iterations):
        padded = F.pad(result, (1, 1, 1, 1), mode="replicate")
        neighbors = (padded[:, :, :-2, 1:-1], padded[:, :, 2:, 1:-1], padded[:, :, 1:-1, :-2], padded[:, :, 1:-1, 2:])
        update = sum(gate * torch.exp(-((neighbor - result) / kappa).square()) * (neighbor - result) for neighbor in neighbors)
        result = result + step_size * update
    return _normalize(result)


class RDIFCAM:
    """Generate RDIF-CAM maps from a PyTorch model and a target convolution layer."""

    def __init__(self, model, target_layer, ig_steps=40, diffusion_iterations=15, kappa=0.20, step_size=0.25):
        self.model = model
        self.target_layer = target_layer
        self.ig_steps = ig_steps
        self.diffusion_iterations = diffusion_iterations
        self.kappa = kappa
        self.step_size = step_size

    def __call__(self, image, target_class):
        if image.shape[0] != 1:
            raise ValueError("RDIFCAM currently accepts a batch size of one.")
        activations, gradients = {}, {}
        forward = self.target_layer.register_forward_hook(lambda _, __, output: activations.update(value=output))
        backward = self.target_layer.register_full_backward_hook(lambda _, __, output: gradients.update(value=output[0]))
        baseline, accumulated = torch.zeros_like(image), None
        try:
            for index in range(self.ig_steps + 1):
                sample = (baseline + index / self.ig_steps * (image - baseline)).detach().requires_grad_(True)
                output = self.model(sample)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                self.model.zero_grad(set_to_none=True)
                output[:, target_class].sum().backward()
                accumulated = gradients["value"].detach() if accumulated is None else accumulated + gradients["value"].detach()
            weights = (accumulated / (self.ig_steps + 1)).abs().mean(dim=(-2, -1), keepdim=True)
            seed = F.relu((weights * activations["value"].detach()).sum(dim=1, keepdim=True))
            seed = F.interpolate(seed, image.shape[-2:], mode="bilinear", align_corners=False)
            return _diffuse(_normalize(seed), radiomic_gate(image), self.diffusion_iterations, self.kappa, self.step_size)
        finally:
            forward.remove()
            backward.remove()
