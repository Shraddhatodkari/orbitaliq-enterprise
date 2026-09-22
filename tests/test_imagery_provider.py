import base64
import io

import httpx
import numpy as np
import pytest
import respx
from PIL import Image

from orbitaliq.config import Settings
from orbitaliq.data.imagery_provider import (
    DEFAULT_ZOOM,
    ChangeTilePairResult,
    ImageryResult,
    NasaGibsImageryProvider,
    approximate_resolution_m_per_pixel,
    build_satellite_visual_package,
    diff_heatmap_png_data_uri,
    tile_to_png_data_uri,
    _lonlat_to_tile_xy,
)
from orbitaliq.data.synthetic_data import TILE_SIZE


@pytest.fixture
def settings():
    return Settings(ORBITALIQ_LIVE_DATA_MODE=True, LIVE_DATA_TIMEOUT_SECONDS=2.0)


def _fake_jpeg_bytes(size=(256, 256), color=(40, 70, 130)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "lat,lon,zoom",
    [(0.0, 0.0, 0), (35.6762, 139.6503, 6), (-33.87, 151.21, 8), (89.9, 179.9, 5), (-89.9, -179.9, 5)],
)
def test_lonlat_to_tile_xy_stays_in_grid_bounds(lat, lon, zoom):
    x, y = _lonlat_to_tile_xy(lat, lon, zoom)
    n = 2**zoom
    assert 0 <= x < n
    assert 0 <= y < n


@respx.mock
def test_fetch_satellite_tile_live_success(settings):
    respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(200, content=_fake_jpeg_bytes(), headers={"content-type": "image/jpeg"})
    )
    provider = NasaGibsImageryProvider(settings=settings)
    result = provider.fetch_satellite_tile(35.6762, 139.6503)

    assert result.live is True
    assert result.source == "nasa_gibs_live"
    assert result.tile.shape == (3, TILE_SIZE, TILE_SIZE)
    assert result.tile.dtype == np.float32
    assert 0.0 <= result.tile.min() and result.tile.max() <= 1.0
    provider.close()


@respx.mock
def test_fetch_satellite_tile_falls_back_on_404(settings):
    respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(404, text="not found")
    )
    provider = NasaGibsImageryProvider(settings=settings)
    result = provider.fetch_satellite_tile(35.6762, 139.6503)

    assert result.live is False
    assert result.source == "insufficient_data"
    assert result.status == "INSUFFICIENT_DATA"
    # A 404 (no mosaic published yet for this date) is a documented,
    # expected gap -- INSUFFICIENT_DATA, not ERROR. See
    # core/data_quality_state.py.
    assert result.quality_state == "INSUFFICIENT_DATA"
    assert result.tile.shape == (3, TILE_SIZE, TILE_SIZE)
    # A blank placeholder, never the deterministic synthetic-imagery
    # generator (which is reserved for offline demo mode / test fixtures).
    assert np.array_equal(result.tile, np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32))
    provider.close()


@respx.mock
def test_fetch_satellite_tile_falls_back_on_non_image_content_type(settings):
    respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(200, text="<html>error page</html>", headers={"content-type": "text/html"})
    )
    provider = NasaGibsImageryProvider(settings=settings)
    result = provider.fetch_satellite_tile(0.0, 0.0)

    assert result.live is False
    assert "non-image content-type" in result.summary
    # An unexpected response shape (not a documented "no data for this
    # date" 404) -- ERROR, not INSUFFICIENT_DATA. See
    # core/data_quality_state.py.
    assert result.status == "INSUFFICIENT_DATA"  # coarse flag unchanged
    assert result.quality_state == "ERROR"
    assert result.source == "error:nasa_gibs:ValueError"
    provider.close()


@respx.mock
def test_fetch_satellite_tile_falls_back_on_network_error(settings):
    respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(side_effect=httpx.ConnectError("boom"))
    provider = NasaGibsImageryProvider(settings=settings)
    result = provider.fetch_satellite_tile(0.0, 0.0)

    assert result.live is False
    # An unexpected transport failure -- ERROR, not INSUFFICIENT_DATA.
    assert result.status == "INSUFFICIENT_DATA"  # coarse flag unchanged
    assert result.quality_state == "ERROR"
    assert result.source == "error:nasa_gibs:ConnectError"
    provider.close()


@respx.mock
def test_fetch_satellite_tile_uses_custom_date_and_layer(settings):
    route = respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(200, content=_fake_jpeg_bytes(), headers={"content-type": "image/jpeg"})
    )
    provider = NasaGibsImageryProvider(settings=settings)
    result = provider.fetch_satellite_tile(
        35.6762, 139.6503, tile_date="2026-01-15", layer="MODIS_Terra_CorrectedReflectance_TrueColor"
    )
    assert route.called
    requested_url = str(route.calls.last.request.url)
    assert "2026-01-15" in requested_url
    assert "MODIS_Terra_CorrectedReflectance_TrueColor" in requested_url
    assert result.live is True
    provider.close()


def test_provenance_dict_shape():
    from orbitaliq.data.imagery_provider import ImageryResult

    result = ImageryResult(tile=np.zeros((3, 4, 4), dtype=np.float32), live=True, source="x", summary="y")
    assert result.provenance() == {
        "live": True, "source": "x", "summary": "y", "status": "AVAILABLE", "quality_state": "LIVE",
    }


@respx.mock
def test_fetch_change_pair_live_success_requests_two_distinct_dates(settings):
    route = respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(200, content=_fake_jpeg_bytes(), headers={"content-type": "image/jpeg"})
    )
    provider = NasaGibsImageryProvider(settings=settings)
    pair = provider.fetch_change_pair(35.6762, 139.6503, lookback_days=180)

    assert route.call_count == 2
    # URL shape: .../default/{tile_date}/{tile_matrix_set}/{zoom}/{y}/{x}.jpg
    requested_dates = {str(call.request.url).split("/")[-5] for call in route.calls}
    assert len(requested_dates) == 2  # current and prior dates differ

    assert pair.live is True
    assert pair.source == "nasa_gibs_live"
    assert pair.current.tile.shape == (3, TILE_SIZE, TILE_SIZE)
    assert pair.prior.tile.shape == (3, TILE_SIZE, TILE_SIZE)
    assert pair.lookback_days == 180
    provider.close()


@respx.mock
def test_fetch_change_pair_reports_insufficient_data_never_fabricated_imagery_on_failure(settings):
    """When NASA GIBS is fully unreachable in live mode, both halves of the
    pair must come back as a typed INSUFFICIENT_DATA result with a blank
    placeholder tile -- never the deterministic synthetic-imagery generator
    (reserved for offline demo mode), and never two different-looking
    fabricated tiles standing in for a real before/after story.
    """
    respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg").mock(
        return_value=httpx.Response(404, text="not found")
    )
    provider = NasaGibsImageryProvider(settings=settings)
    pair = provider.fetch_change_pair(10.0, 20.0)

    assert pair.live is False
    assert pair.source == "insufficient_data"
    assert pair.current.status == "INSUFFICIENT_DATA"
    assert pair.prior.status == "INSUFFICIENT_DATA"
    # Both are the same blank placeholder -- no fabricated "plausible story".
    assert np.array_equal(pair.current.tile, pair.prior.tile)
    assert np.array_equal(pair.current.tile, np.zeros_like(pair.current.tile))
    provider.close()


@respx.mock
def test_fetch_change_pair_partial_live_is_disclosed(settings):
    route = respx.get(url__regex=r"https://gibs\.earthdata\.nasa\.gov/wmts/.*\.jpg")
    route.side_effect = [
        httpx.Response(200, content=_fake_jpeg_bytes(), headers={"content-type": "image/jpeg"}),
        httpx.Response(404, text="not found"),
    ]
    provider = NasaGibsImageryProvider(settings=settings)
    pair = provider.fetch_change_pair(10.0, 20.0)

    assert pair.current.live is True
    assert pair.prior.live is False
    assert pair.live is False
    assert pair.source == "nasa_gibs_partial_live"
    provider.close()


# --- Resolution honesty (approximate_resolution_m_per_pixel) ---


def test_resolution_is_computed_not_a_fixed_marketing_number():
    equator = approximate_resolution_m_per_pixel(0.0, zoom=DEFAULT_ZOOM)
    high_lat = approximate_resolution_m_per_pixel(60.0, zoom=DEFAULT_ZOOM)
    # Web-Mercator: resolution improves (smaller m/pixel) moving away from
    # the equator at a fixed zoom -- a genuine geometric computation, not a
    # constant.
    assert high_lat < equator
    # Within the claimed ~600m/pixel (60N/S) to ~1.2km/pixel (equator) range.
    assert 500 < high_lat < 700
    assert 1000 < equator < 1300


def test_resolution_scales_with_zoom():
    coarse = approximate_resolution_m_per_pixel(30.0, zoom=6)
    fine = approximate_resolution_m_per_pixel(30.0, zoom=9)
    assert fine < coarse


# --- PNG rendering helpers ---


def test_tile_to_png_data_uri_produces_valid_png():
    tile = np.random.rand(3, 8, 8).astype(np.float32)
    uri = tile_to_png_data_uri(tile)
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"
    assert img.size == (8, 8)


def test_tile_to_png_data_uri_clips_out_of_range_values():
    tile = np.full((3, 4, 4), 5.0, dtype=np.float32)  # out of [0, 1]
    uri = tile_to_png_data_uri(tile)  # must not raise
    assert uri.startswith("data:image/png;base64,")


def test_diff_heatmap_png_data_uri_produces_valid_png():
    current = np.full((3, 8, 8), 0.8, dtype=np.float32)
    prior = np.full((3, 8, 8), 0.2, dtype=np.float32)
    uri = diff_heatmap_png_data_uri(current, prior)
    raw = base64.b64decode(uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"
    assert img.size == (8, 8)


def test_diff_heatmap_handles_identical_tiles_without_dividing_by_zero():
    same = np.full((3, 4, 4), 0.5, dtype=np.float32)
    uri = diff_heatmap_png_data_uri(same, same)  # must not raise / NaN
    assert uri.startswith("data:image/png;base64,")


# --- build_satellite_visual_package ---


def _imagery_result(*, live, source, date, lat=39.5, lon=-119.4) -> ImageryResult:
    return ImageryResult(
        tile=np.random.rand(3, 64, 64).astype(np.float32),
        live=live, source=source, summary="test",
        observation_date=date, latitude=lat, longitude=lon,
        layer="VIIRS_SNPP_CorrectedReflectance_TrueColor", zoom=DEFAULT_ZOOM,
        resolution_m_per_pixel=approximate_resolution_m_per_pixel(lat, DEFAULT_ZOOM),
    )


def test_build_satellite_visual_package_shape_and_provenance():
    pair = ChangeTilePairResult(
        current=_imagery_result(live=True, source="nasa_gibs_live", date="2026-09-15"),
        prior=_imagery_result(live=True, source="nasa_gibs_live", date="2026-03-15"),
        lookback_days=180,
    )
    package = build_satellite_visual_package(pair)

    assert package["current_observation_date"] == "2026-09-15"
    assert package["prior_observation_date"] == "2026-03-15"
    assert package["lookback_days"] == 180
    assert package["current_image_available"] is True
    assert package["prior_image_available"] is True
    assert package["current_provenance"] == {
        "live": True, "source": "nasa_gibs_live", "summary": "test", "status": "AVAILABLE", "quality_state": "LIVE",
    }
    assert package["resolution_m_per_pixel"] == pytest.approx(approximate_resolution_m_per_pixel(39.5, DEFAULT_ZOOM))
    assert package["coordinates"] == {"latitude": 39.5, "longitude": -119.4}
    assert package["current_image_png"].startswith("data:image/png;base64,")
    assert package["prior_image_png"].startswith("data:image/png;base64,")
    assert package["diff_image_png"].startswith("data:image/png;base64,")
    assert "not confirmation of any specific business event" in package["change_score_note"]


def test_build_satellite_visual_package_discloses_non_live_fallback_honestly():
    pair = ChangeTilePairResult(
        current=_imagery_result(live=False, source="insufficient_data", date="2026-09-15"),
        prior=_imagery_result(live=False, source="insufficient_data", date="2026-03-15"),
        lookback_days=180,
    )
    package = build_satellite_visual_package(pair)
    assert package["current_image_available"] is False
    assert package["prior_image_available"] is False
    assert package["current_provenance"]["source"] == "insufficient_data"
