from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class UpscaleJob:
    factor: int
    weights_path: Path
    engine: str = "real_esrgan"
    tile_size: int = 512
    tile_pad: int = 16


class UpscalerEngine(Protocol):
    id: str

    def upscale(
        self,
        image: Image.Image,
        *,
        job: UpscaleJob,
        settings: Any,
    ) -> Image.Image: ...


@dataclass(frozen=True)
class TileSlice:
    x0: int
    y0: int
    x1: int
    y1: int


def tile_slices(width: int, height: int, tile_size: int) -> list[TileSlice]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if tile_size <= 0 or tile_size >= max(width, height):
        return [TileSlice(0, 0, width, height)]

    tiles: list[TileSlice] = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            tiles.append(TileSlice(x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)))
    return tiles


def create_upscaler_engine(engine: str | None = None) -> UpscalerEngine:
    engine_id = (engine or "real_esrgan").strip().lower().replace("-", "_")
    if engine_id in {"real_esrgan", "realesrgan"}:
        return RealESRGANUpscaler()
    if engine_id in {"aura_sr", "aurasr"}:
        return AuraSRUpscaler()
    raise RuntimeError(f"Unsupported upscaler engine: {engine}.")


class RealESRGANUpscaler:
    id = "real_esrgan"

    def __init__(self) -> None:
        # Cache the loaded network per (weights, factor) so a multi-image batch
        # loads the weights once. device/dtype are stable within a batch (the
        # asset writer holds one engine instance per job), so they ride along.
        self._models: dict[tuple[str, int], tuple[Any, str, Any]] = {}

    def load(self, job: UpscaleJob, settings: Any) -> str:
        """Load + cache the network for ``job`` and return the resolved device.

        Lets a caller preload (and surface load telemetry) before the first
        image, then reuse the cached network across the rest of the batch.
        """
        _model, device, _dtype = self._prepare(importlib.import_module("torch"), job, settings)
        return device

    def upscale(
        self,
        image: Image.Image,
        *,
        job: UpscaleJob,
        settings: Any,
    ) -> Image.Image:
        torch = importlib.import_module("torch")
        model, device, dtype = self._prepare(torch, job, settings)
        return self._upscale_with_model(
            torch,
            model,
            image.convert("RGB"),
            factor=job.factor,
            device=device,
            dtype=dtype,
            tile_size=job.tile_size,
            tile_pad=job.tile_pad,
        )

    def _prepare(self, torch: Any, job: UpscaleJob, settings: Any) -> tuple[Any, str, Any]:
        if job.factor not in {2, 4}:
            raise RuntimeError("Real-ESRGAN upscaling supports only 2x and 4x factors.")
        if not job.weights_path.exists():
            raise RuntimeError(f"Real-ESRGAN weights are missing: {job.weights_path}")
        cache_key = (str(job.weights_path), job.factor)
        cached = self._models.get(cache_key)
        if cached is not None:
            return cached

        from .image_adapters import activate_torch_device, select_torch_device, select_torch_dtype

        device = select_torch_device(torch, getattr(settings, "gpu_id", None))
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, None)
        model = self._load_model(torch, job.weights_path, factor=job.factor, device=device, dtype=dtype)
        prepared = (model, device, dtype)
        self._models[cache_key] = prepared
        return prepared

    def _load_model(
        self,
        torch: Any,
        weights_path: Path,
        *,
        factor: int,
        device: str,
        dtype: Any,
    ) -> Any:
        state = _load_state_dict(torch, weights_path)
        num_blocks = _infer_rrdb_blocks(state)
        RRDBNet = _rrdbnet_class(torch)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_blocks, num_grow_ch=32, scale=factor)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Real-ESRGAN weights are missing {len(missing)} network tensors.")
        if unexpected:
            raise RuntimeError(f"Real-ESRGAN weights contain {len(unexpected)} unexpected tensors.")
        model.eval()
        return model.to(device=device, dtype=dtype)

    def _upscale_with_model(
        self,
        torch: Any,
        model: Any,
        image: Image.Image,
        *,
        factor: int,
        device: str,
        dtype: Any,
        tile_size: int,
        tile_pad: int,
    ) -> Image.Image:
        return _run_tiled(torch, model, image, factor=factor, device=device, dtype=dtype, tile_size=tile_size, tile_pad=tile_pad)


class AuraSRUpscaler:
    id = "aura_sr"

    def upscale(
        self,
        image: Image.Image,
        *,
        job: UpscaleJob,
        settings: Any,
    ) -> Image.Image:
        if job.factor != 4:
            raise RuntimeError("AuraSR upscaling supports only 4x output.")
        if not job.weights_path.exists():
            raise RuntimeError(f"AuraSR weights are missing: {job.weights_path}")

        torch = importlib.import_module("torch")
        aura_sr = importlib.import_module("aura_sr")
        from .image_adapters import activate_torch_device, select_torch_device

        device = select_torch_device(torch, getattr(settings, "gpu_id", None))
        activate_torch_device(torch, device)
        model = aura_sr.AuraSR.from_pretrained(str(job.weights_path), use_safetensors=True)
        upsampler = getattr(model, "upsampler", None)
        if upsampler is not None and hasattr(upsampler, "to"):
            upsampler.to(device)
        if upsampler is not None and hasattr(upsampler, "eval"):
            upsampler.eval()
        if job.tile_pad > 0 and hasattr(model, "upscale_4x_overlapped"):
            return model.upscale_4x_overlapped(image.convert("RGB"))
        return model.upscale_4x(image.convert("RGB"))


def _load_state_dict(torch: Any, weights_path: Path) -> dict[str, Any]:
    if weights_path.suffix.lower() == ".safetensors":
        safetensors_torch = importlib.import_module("safetensors.torch")
        checkpoint = safetensors_torch.load_file(str(weights_path), device="cpu")
    else:
        try:
            checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(weights_path), map_location="cpu")

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise RuntimeError("Real-ESRGAN weights did not contain a state dict.")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _infer_rrdb_blocks(state: dict[str, Any]) -> int:
    block_indexes = []
    for key in state:
        match = re.match(r"body\.(\d+)\.", key)
        if match:
            block_indexes.append(int(match.group(1)))
    return max(block_indexes) + 1 if block_indexes else 23


def _rrdbnet_class(torch: Any) -> type:
    nn = torch.nn
    F = torch.nn.functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x: Any) -> Any:
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x: Any) -> Any:
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(
            self,
            *,
            num_in_ch: int,
            num_out_ch: int,
            scale: int,
            num_feat: int = 64,
            num_block: int = 23,
            num_grow_ch: int = 32,
        ) -> None:
            super().__init__()
            self.scale = scale
            if scale == 2:
                num_in_ch *= 4
            elif scale == 1:
                num_in_ch *= 16
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x: Any) -> Any:
            if self.scale == 2:
                feat = F.pixel_unshuffle(x, 2)
            elif self.scale == 1:
                feat = F.pixel_unshuffle(x, 4)
            else:
                feat = x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet


def _run_tiled(
    torch: Any,
    model: Any,
    image: Image.Image,
    *,
    factor: int,
    device: str,
    dtype: Any,
    tile_size: int,
    tile_pad: int,
) -> Image.Image:
    width, height = image.size
    output = torch.empty((1, 3, height * factor, width * factor), dtype=torch.float32, device="cpu")
    context = getattr(torch, "inference_mode", None) or getattr(torch, "no_grad")
    with context():
        for tile in tile_slices(width, height, tile_size):
            crop_x0 = max(0, tile.x0 - tile_pad)
            crop_y0 = max(0, tile.y0 - tile_pad)
            crop_x1 = min(width, tile.x1 + tile_pad)
            crop_y1 = min(height, tile.y1 + tile_pad)
            tile_image = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            tile_tensor = _image_to_tensor(torch, tile_image, device=device, dtype=dtype)
            tile_output = model(tile_tensor)

            inner_x0 = (tile.x0 - crop_x0) * factor
            inner_y0 = (tile.y0 - crop_y0) * factor
            inner_x1 = inner_x0 + (tile.x1 - tile.x0) * factor
            inner_y1 = inner_y0 + (tile.y1 - tile.y0) * factor
            output[
                ...,
                tile.y0 * factor : tile.y1 * factor,
                tile.x0 * factor : tile.x1 * factor,
            ] = tile_output[..., inner_y0:inner_y1, inner_x0:inner_x1].detach().float().cpu()
    return _tensor_to_image(torch, output)


def _image_to_tensor(torch: Any, image: Image.Image, *, device: str, dtype: Any) -> Any:
    np = importlib.import_module("numpy")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _tensor_to_image(torch: Any, tensor: Any) -> Image.Image:
    np = importlib.import_module("numpy")
    array = (
        torch.clamp(tensor.squeeze(0), 0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array.astype(np.uint8), "RGB")
