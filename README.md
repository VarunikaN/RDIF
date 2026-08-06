# RDIF-CAM

RDIF-CAM is a post-hoc explainability method for dense prediction models. It creates an Integrated Gradients CAM seed, derives a radiomic texture gate from Gabor, local binary pattern, and local-variance features, then refines the seed with radiomic-guided Perona-Malik diffusion.

The implementation is designed as a compact, reusable module that integrates with trained segmentation networks, including LBA-Net.

## Install

```bash
pip install -r requirements.txt
```

## Use With A PyTorch Model

Select a convolutional layer whose feature maps should be explained. The model must return a tensor shaped `[batch, classes, height, width]` (or return it as the first item of a tuple).

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("src").resolve()))
from core import RDIFCAM
from visualization import save_visualizations

model = ...  # Loaded segmentation model in evaluation mode
target_layer = model.decoder.final_conv
image = ...  # Float tensor with shape [1, channels, height, width]

explainer = RDIFCAM(model, target_layer, ig_steps=40, diffusion_iterations=15)
cam = explainer(image, target_class=1)  # [1, 1, height, width], normalized to [0, 1]
save_visualizations(image, cam, "outputs")
```

## LBA-Net Example

Install [LBA-Net](https://github.com/VarunikaN/LBANet) and this repository in the same environment. The example below loads an LBA-Net checkpoint and explains the final student decoder convolution.

```python
from pathlib import Path
import sys
import torch

sys.path.extend([str(Path("RDIF/src").resolve()), str(Path("LBANet").resolve())])
from core import RDIFCAM
from visualization import save_visualizations
from src.models.student_teacher_unet import StudentTeacherUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentTeacherUNet(num_classes=2, pretrained=False).to(device).eval()
checkpoint = torch.load("path/to/best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

target_layer = next(layer for layer in reversed(list(model.student.decoder1.modules())) if isinstance(layer, torch.nn.Conv2d))
image = torch.rand(1, 3, 428, 428, device=device)  # Apply the same preprocessing used for LBA-Net training.
cam = RDIFCAM(model, target_layer)(image, target_class=1)
save_visualizations(image, cam, "outputs")
```

## Outputs

Use a simple project layout for each experiment:

```text
your_experiment/
├── input/     # Place the image to explain here
└── outputs/   # RDIF-CAM results are written here
```

After loading your model and image from `input/`, call `save_visualizations(image, cam, "outputs")`. It writes only the derived RDIF-CAM artifacts:

```text
outputs/
├── heatmap.png     # RDIF-CAM heatmap
├── overlay.png     # Heatmap blended with the input
└── comparison.png  # Input, heatmap, and overlay side by side
```

Pass `prefix="example_name"` to use a custom output prefix.

## Method

1. Compute channel weights from gradients integrated along a zero-baseline path.
2. Form and upsample the positive weighted activation seed.
3. Build a radiomic gate from multi-orientation Gabor responses, LBP texture, and inverse local variance.
4. Diffuse the seed for 15 iterations using a Perona-Malik edge-stopping function modulated by the radiomic gate.

Defaults follow the RDIF project configuration: 40 IG steps, 15 diffusion iterations, `kappa=0.20`, and diffusion step size `0.25`.
