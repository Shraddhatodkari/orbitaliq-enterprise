"""Small, dependency-free geospatial helpers used across the pipeline."""
from __future__ import annotations

import hashlib
import math


def validate_lat_lon(latitude: float, longitude: float) -> None:
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude {latitude} out of range [-90, 90]")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude {longitude} out of range [-180, 180]")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two lat/lon points."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def deterministic_seed(*parts: str | float) -> int:
    """Stable integer seed derived from arbitrary inputs.

    Used to make synthetic-data generation and vision inference
    reproducible per-region without relying on global RNG state — the same
    coordinates always yield the same synthetic tile and, absent new
    real-world data, the same risk read-out.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)
