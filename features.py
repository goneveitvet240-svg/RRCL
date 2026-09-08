import os

import numpy as np
import torch


def resolve_device(device=None):
    """Resolve an explicit/env device, then prefer CUDA, MPS, and CPU."""

    requested = device or os.environ.get("RRCL_DEVICE")
    if requested and requested != "auto":
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class DinoFeatureExtractor:
    def __init__(self, backbone="dinov2_vitb14", img_size=518, device=None):
        self.backbone = backbone
        self.img_size = int(img_size)
        self.device = resolve_device(device)
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("Missing timm. Install with: pip install timm") from e
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, image):
        x = torch.from_numpy(np.asarray(image).copy()).float()
        if x.ndim == 2:
            x = x[:, :, None].repeat(1, 1, 3)
        if x.shape[-1] == 4:
            x = x[:, :, :3]
        x = x.permute(2, 0, 1)[None] / 255.0
        x = torch.nn.functional.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            tokens = out.get("x_norm_patchtokens")
            if tokens is None:
                tokens = out.get("x_norm_clstoken")
            if tokens is None:
                raise RuntimeError(f"Unsupported timm DINO output keys: {list(out.keys())}")
        else:
            tokens = out
            if tokens.ndim == 3 and tokens.shape[1] > 1:
                tokens = tokens[:, 1:, :]
        tokens = tokens.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return tokens


def cache_key(image_path, backbone, img_size):
    safe = os.path.abspath(image_path).replace(os.sep, "__").replace(":", "")
    return f"{backbone}_img{int(img_size)}_{safe}.npy"
