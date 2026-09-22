from __future__ import annotations

from fastapi import APIRouter

from orbitaliq.api.schemas import HealthResponse
from orbitaliq.config import get_settings
from orbitaliq.nvidia.vision_model import resolve_device

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    device = resolve_device(settings.orbitaliq_vision_device)
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.orbitaliq_env,
        offline_mode=settings.orbitaliq_offline_mode,
        live_data_mode=settings.orbitaliq_live_data_mode,
        vision_device=str(device),
    )
