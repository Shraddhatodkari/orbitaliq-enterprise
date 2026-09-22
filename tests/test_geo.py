import pytest

from orbitaliq.utils.geo import deterministic_seed, haversine_km, validate_lat_lon


def test_validate_lat_lon_accepts_valid_coordinates():
    validate_lat_lon(0.0, 0.0)
    validate_lat_lon(-90.0, -180.0)
    validate_lat_lon(90.0, 180.0)


@pytest.mark.parametrize(
    "lat,lon",
    [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)],
)
def test_validate_lat_lon_rejects_out_of_range(lat, lon):
    with pytest.raises(ValueError):
        validate_lat_lon(lat, lon)


def test_haversine_km_same_point_is_zero():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_known_distance_ny_to_london():
    # New York (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    dist = haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5400 < dist < 5750


def test_deterministic_seed_is_stable_and_input_sensitive():
    s1 = deterministic_seed(1.0, 2.0, "a")
    s2 = deterministic_seed(1.0, 2.0, "a")
    s3 = deterministic_seed(1.0, 2.0, "b")
    assert s1 == s2
    assert s1 != s3
    assert isinstance(s1, int)
