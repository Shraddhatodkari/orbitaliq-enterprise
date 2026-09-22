"""Vision Agent — runs the NVIDIA-GPU-ready CNN + change-index pipeline
over a bi-temporal satellite tile pair and returns structured
land-cover/construction change findings.
"""
from __future__ import annotations

import numpy as np

from orbitaliq.core.logging_config import logger
from orbitaliq.nvidia.vision_model import VisionFindings, VisionInferenceEngine


class VisionAgent:
    def __init__(self, engine: VisionInferenceEngine) -> None:
        self._engine = engine

    def run(self, current_tile: np.ndarray, prior_tile: np.ndarray) -> VisionFindings:
        findings = self._engine.analyze_change_pair(current_tile, prior_tile)
        logger.debug(
            f"VisionAgent: overall_change_score={findings.overall_change_score:.3f} "
            f"device={findings.device_used}"
        )
        return findings
