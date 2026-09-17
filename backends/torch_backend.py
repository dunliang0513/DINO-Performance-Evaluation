"""PyTorch feature extractors for DINOv2 (transformers) and DINOv3 (timm)."""

import cv2
import numpy as np
import torch

from backends.base import l2_normalise, pool_tokens

# DINOv3 is gated under facebook/*; these timm mirrors carry the same weights
# and are not gated, so no manual approval is needed to reproduce results.
MODEL_SOURCES = {
    "dinov2-small": ("transformers", "facebook/dinov2-small"),
    "dinov2-base": ("transformers", "facebook/dinov2-base"),
    "dinov3-small": ("timm", "vit_small_patch16_dinov3.lvd1689m"),
    "dinov3-base": ("timm", "vit_base_patch16_dinov3.lvd1689m"),
}


class TorchDinoBackend:
    """Embeds BGR crops with a frozen DINO backbone."""

    def __init__(self, name, device="cuda", use_fp16=True):
        if name not in MODEL_SOURCES:
            raise ValueError(
                f"Unknown model {name!r}. "
                f"Expected one of: {', '.join(sorted(MODEL_SOURCES))}."
            )

        self.name = name
        self.device = torch.device(device)
        # fp16 is a CUDA-only optimisation; on CPU it is slower, not faster.
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        source, model_id = MODEL_SOURCES[name]
        self.source = source
        self.model_id = model_id

        if source == "transformers":
            self._load_transformers(model_id)
        else:
            self._load_timm(model_id)

        self.model.eval()
        self.model.to(device=self.device, dtype=self.dtype)

    def _load_transformers(self, model_id):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.dim = self.model.config.hidden_size
        # DINOv2 in transformers has a single CLS prefix token, no registers.
        self.num_prefix_tokens = 1

    def _load_timm(self, model_id):
        import timm

        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.processor = timm.data.create_transform(
            **data_config, is_training=False
        )
        self.dim = self.model.num_features
        # DINOv3 prepends CLS plus register tokens; timm reports the total.
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)

    def _to_pixel_values(self, crops_bgr):
        """BGR uint8 crops -> a normalised, batched float tensor."""
        from PIL import Image

        images = [
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            for crop in crops_bgr
        ]

        if self.source == "transformers":
            batch = self.processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"]
        else:
            pixel_values = torch.stack([self.processor(image) for image in images])

        return pixel_values.to(device=self.device, dtype=self.dtype)

    def _forward_tokens(self, pixel_values):
        """Return the full token sequence [B, T, D]."""
        if self.source == "transformers":
            return self.model(pixel_values=pixel_values).last_hidden_state

        return self.model.forward_features(pixel_values)

    @torch.inference_mode()
    def embed(self, crops_bgr, pooling="cls"):
        if not crops_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)

        pixel_values = self._to_pixel_values(crops_bgr)
        tokens = self._forward_tokens(pixel_values)

        pooled = pool_tokens(tokens, pooling, self.num_prefix_tokens)
        # Normalise in float32: fp16 norms lose precision near unit length.
        normalised = l2_normalise(pooled.float())

        return normalised.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def warmup(self, batch_size, iterations=5):
        """Run throwaway inferences so later timings exclude lazy CUDA setup."""
        dummy = [
            np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(batch_size)
        ]

        for _ in range(iterations):
            self.embed(dummy, pooling="cls")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
