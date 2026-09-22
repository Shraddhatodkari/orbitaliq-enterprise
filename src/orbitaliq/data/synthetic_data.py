"""Synthetic satellite-tile fallback used when live NASA GIBS imagery is
unavailable (outage, rate limit, no cloud-free pass for the requested
date) or when ``ORBITALIQ_LIVE_DATA_MODE=false`` (offline demo mode, used
by the test suite/CI so it never depends on network availability).

The generator is deterministic per-(lat, lon, seed_salt): the same site and
tile "role" (``"current"`` vs ``"prior"``) always yields the same tile,
which keeps demos and tests reproducible without a network call. The
underlying low-frequency terrain texture is shared across both tiles for a
given site (same coordinates -> same base terrain), while a
``"current"``-salted tile additionally gets an injected bright,
low-vegetation patch simulating a newly cleared/built footprint -- so
even the clearly-labeled synthetic fallback tells a plausible before/after
story for the change-detection demo, rather than two unrelated noise
fields.
"""
from __future__ import annotations

import numpy as np

from orbitaliq.utils.geo import deterministic_seed

TILE_SIZE = 64  # pixels, square tile
TILE_CHANNELS = 3  # RGB


def generate_satellite_tile(latitude: float, longitude: float, *, seed_salt: str = "tile") -> np.ndarray:
    """Generate a synthetic RGB satellite tile as a float32 array in
    ``[C, H, W]`` layout (matching PyTorch convention), values in ``[0, 1]``.

    ``seed_salt`` distinguishes tile "roles" at the same coordinates (e.g.
    ``"current"`` vs ``"prior"``) while keeping the shared base terrain
    consistent, so a synthetic-fallback change-detection pair still looks
    like two passes over the same site rather than two different places.
    """
    terrain_seed = deterministic_seed(round(latitude, 3), round(longitude, 3), "terrain")
    rng_terrain = np.random.default_rng(terrain_seed)
    coarse = rng_terrain.normal(loc=0.42, scale=0.10, size=(TILE_CHANNELS, 8, 8)).astype(np.float32)
    tile = np.kron(coarse, np.ones((1, TILE_SIZE // 8, TILE_SIZE // 8), dtype=np.float32))

    variant_seed = deterministic_seed(round(latitude, 3), round(longitude, 3), seed_salt)
    rng = np.random.default_rng(variant_seed)
    tile = tile + rng.normal(loc=0.0, scale=0.03, size=tile.shape).astype(np.float32)

    if seed_salt.startswith("current"):
        # Bare-earth/concrete signature: brighter and less green than
        # surrounding vegetation/terrain -- the same spectral cue real
        # bi-temporal change-detection heuristics key off.
        strength = float(rng.uniform(0.35, 0.85))
        patch_size = min(int(10 + strength * 28), TILE_SIZE)
        top = int(rng.integers(0, TILE_SIZE - patch_size + 1))
        left = int(rng.integers(0, TILE_SIZE - patch_size + 1))
        tile[:, top : top + patch_size, left : left + patch_size] += 0.22 * strength
        tile[1, top : top + patch_size, left : left + patch_size] -= 0.15 * strength

    return np.clip(tile, 0.0, 1.0).astype(np.float32)
