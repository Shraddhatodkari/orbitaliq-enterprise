"""GPU-accelerated satellite-imagery change detection for competitive
intelligence.

This module is written to run unchanged on an NVIDIA GPU (e.g. an
A10 / L4 / T4 / H100 in production) or transparently fall back to CPU (e.g.
this sandbox, a laptop, or a CI runner) via :func:`resolve_device`.

**What this honestly can and cannot see.** The imagery behind this pipeline
(``data/imagery_provider.py``, NASA GIBS VIIRS/MODIS) is ~250m-1km per
pixel natively, and this client's own resize to a fixed 64x64px tile
coarsens that further to an *effective* ~600m/pixel-1.2km/pixel depending
on latitude — see ``imagery_provider.approximate_resolution_m_per_pixel()``,
which every satellite API response reports as an actual computed number,
not a marketing spec. That resolution can show large-scale land-cover change at a facility
— a new building footprint, a cleared lot, a large parking/laydown-yard
expansion — which is exactly the scale of signal real alternative-data
analysts used to track Tesla's Gigafactory Nevada build-out via satellite
through 2018-2020. It cannot resolve individual vehicles or small
structures; that level of detail needs commercial sub-meter imagery
(Planet, Maxar), which is not free and is not what this project uses. Every
finding below is scoped to "large-scale visual change," never anything
finer.

Two complementary signal paths feed every prediction, run over a genuine
**bi-temporal pair** of tiles (current vs. ``lookback_days`` earlier — see
:meth:`~orbitaliq.data.imagery_provider.NasaGibsImageryProvider.fetch_change_pair`)
rather than classifying a single snapshot:

1. **Learned path** — :class:`FacilityChangeCNN`, a small PyTorch CNN over
   the stacked 6-channel (current RGB + prior RGB) tile pair, designed to
   be fine-tuned on labeled bi-temporal change-detection imagery (e.g. the
   xView2 or OSCD change-detection datasets) and exported to ONNX /
   TensorRT for low-latency GPU serving (see :func:`export_to_onnx`).
   Shipped here with deterministic, seeded (untrained) weights, since no
   proprietary labeled imagery ships with this repository.
2. **Analytic / change-index path** — :func:`compute_change_indices`,
   classic remote-sensing band-difference heuristics (brightness delta as
   a bare-earth/concrete proxy, greenness loss as a vegetation-clearing
   proxy, pixel-level change energy). This is the same family of technique
   used operationally before a labeled fine-tuning set exists, so the
   system produces meaningful, testable signal out of the box instead of
   noise from an untrained network.

:class:`VisionInferenceEngine.analyze_change_pair` blends both paths; once
a fine-tuned checkpoint is supplied via ``VISION_MODEL_CHECKPOINT`` the CNN
weight in the blend should be increased (see ``cnn_weight``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from orbitaliq.core.logging_config import logger

DeviceChoice = Literal["auto", "cuda", "cpu"]


def resolve_device(preference: DeviceChoice = "auto") -> torch.device:
    """Pick the compute device the same way in dev, CI, and production.

    ``"auto"`` uses an NVIDIA GPU when CUDA is visible to PyTorch and
    silently falls back to CPU otherwise, so this code path never needs to
    branch by environment.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FacilityChangeCNN(nn.Module):
    """Compact CNN over a stacked 64x64 current+prior RGB tile pair (6
    input channels — early fusion of the two acquisitions).

    Three sigmoid outputs: [construction_expansion_probability,
    vegetation_clearing_probability, overall_change_probability].
    Deliberately small (~90K params) so it trains fast on a single GPU and
    exports cleanly to ONNX/TensorRT for low-latency batch inference over
    large tile grids.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (64, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class VisionFindings:
    # float | None: None exactly when status == "INSUFFICIENT_DATA" (no
    # verified facility coordinates -- see agents/orchestrator.py, which
    # never fabricates a tile pair or a change score to fill this gap; it
    # simply never invokes vision inference at all in that case). Every
    # real vision-engine run always populates all four as floats -- this
    # is purely additive and does not change existing behavior.
    construction_expansion_signal: float | None
    vegetation_clearing_signal: float | None
    overall_change_score: float | None
    change_energy: float | None
    device_used: str
    model_finetuned: bool
    # "AVAILABLE" | "INSUFFICIENT_DATA" -- mirrors FinancialMetric's status
    # contract (see data/sec_edgar_client.py). Defaults to "AVAILABLE" so
    # every existing construction of this dataclass (which never passed
    # this field) is unaffected.
    status: str = "AVAILABLE"

    def as_dict(self) -> dict:
        def _r(v: float | None) -> float | None:
            return round(v, 4) if v is not None else None

        return {
            "construction_expansion_signal": _r(self.construction_expansion_signal),
            "vegetation_clearing_signal": _r(self.vegetation_clearing_signal),
            "overall_change_score": _r(self.overall_change_score),
            "change_energy": _r(self.change_energy),
            "device_used": self.device_used,
            "model_finetuned": self.model_finetuned,
            "status": self.status,
        }


def compute_change_indices(current_tile: np.ndarray, prior_tile: np.ndarray) -> dict:
    """Band-difference remote-sensing heuristics between two ``[C, H, W]``
    RGB tiles of the same site.

    Real deployments would compute these from actual multispectral bands
    (NIR/SWIR for a true NDVI difference); here R/G/B stand in as a proxy
    so the signal is illustrative without requiring a multispectral data
    source. This is deliberately a *large-scale* heuristic (whole-tile
    means), matching the ~250m-1km/pixel resolution of the imagery it runs
    on — see the module docstring.
    """
    if current_tile.shape != prior_tile.shape or current_tile.ndim != 3 or current_tile.shape[0] != 3:
        raise ValueError(f"expected matching (3, H, W) tiles, got {current_tile.shape} vs {prior_tile.shape}")

    diff = current_tile - prior_tile
    brightness_delta = float(current_tile.mean() - prior_tile.mean())
    greenness_delta = float(current_tile[1].mean() - prior_tile[1].mean())

    # New bare-earth/concrete tends to be brighter and less green than the
    # vegetation or terrain it replaced.
    construction_index = float(np.clip(brightness_delta * 2.5 + 0.15, 0.0, 1.0))
    clearing_index = float(np.clip(-greenness_delta * 3.0 + 0.15, 0.0, 1.0))
    change_energy = float(np.clip(diff.std() * 4.0, 0.0, 1.0))

    return {
        "construction_index": construction_index,
        "clearing_index": clearing_index,
        "change_energy": change_energy,
    }


class VisionInferenceEngine:
    """Loads the CNN once per process and serves change-detection inference."""

    def __init__(
        self,
        device_preference: DeviceChoice = "auto",
        checkpoint_path: str | None = None,
        cnn_weight: float = 0.35,
        seed: int = 42,
    ) -> None:
        self.device = resolve_device(device_preference)
        torch.manual_seed(seed)
        self.model = FacilityChangeCNN().to(self.device)
        self.model.eval()
        self.cnn_weight = cnn_weight
        self.model_finetuned = False

        if checkpoint_path and os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model_finetuned = True
            logger.info(f"Loaded fine-tuned vision checkpoint from {checkpoint_path}")
        else:
            logger.info(
                "No fine-tuned checkpoint supplied — vision CNN running with "
                "seeded random initialization; predictions are blended with "
                "the change-index analytic path (see compute_change_indices)."
            )

    @torch.no_grad()
    def analyze_change_pair(self, current_tile: np.ndarray, prior_tile: np.ndarray) -> VisionFindings:
        if current_tile.dtype != np.float32:
            current_tile = current_tile.astype(np.float32)
        if prior_tile.dtype != np.float32:
            prior_tile = prior_tile.astype(np.float32)

        stacked = np.concatenate([current_tile, prior_tile], axis=0)  # (6, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        cnn_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        analytic = compute_change_indices(current_tile, prior_tile)
        w = self.cnn_weight
        construction = w * float(cnn_probs[0]) + (1 - w) * analytic["construction_index"]
        clearing = w * float(cnn_probs[1]) + (1 - w) * analytic["clearing_index"]
        overall = w * float(cnn_probs[2]) + (1 - w) * analytic["change_energy"]

        return VisionFindings(
            construction_expansion_signal=construction,
            vegetation_clearing_signal=clearing,
            overall_change_score=max(construction, clearing, overall),
            change_energy=analytic["change_energy"],
            device_used=str(self.device),
            model_finetuned=self.model_finetuned,
        )

    def analyze_batch(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[VisionFindings]:
        return [self.analyze_change_pair(current, prior) for current, prior in pairs]


def export_to_onnx(model: nn.Module, output_path: str, *, input_shape: tuple[int, int, int] = (6, 64, 64)) -> dict:
    """Export the CNN to ONNX for a TensorRT build step (``trtexec --onnx=...``)
    on the target NVIDIA GPU. Never raises — returns a status dict so
    callers (and tests) can assert on success without requiring the
    optional ``onnx`` package to be installed in every environment.
    """
    dummy = torch.randn(1, *input_shape)
    try:
        model.eval()
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["facility_tile_pair"],
            output_names=["change_logits"],
            dynamic_axes={"facility_tile_pair": {0: "batch"}, "change_logits": {0: "batch"}},
            opset_version=17,
        )
        return {"success": True, "path": output_path, "error": None}
    except Exception as exc:  # pragma: no cover - exercised only when export toolchain is missing
        logger.warning(f"ONNX export skipped/failed: {exc}")
        return {"success": False, "path": None, "error": str(exc)}
