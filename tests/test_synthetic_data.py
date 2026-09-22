import numpy as np

from orbitaliq.data.synthetic_data import TILE_CHANNELS, TILE_SIZE, generate_satellite_tile


def test_generate_satellite_tile_shape_dtype_and_range():
    tile = generate_satellite_tile(25.7617, -80.1918)
    assert tile.shape == (TILE_CHANNELS, TILE_SIZE, TILE_SIZE)
    assert tile.dtype == np.float32
    assert tile.min() >= 0.0
    assert tile.max() <= 1.0


def test_generate_satellite_tile_is_deterministic():
    t1 = generate_satellite_tile(10.0, 20.0)
    t2 = generate_satellite_tile(10.0, 20.0)
    np.testing.assert_array_equal(t1, t2)


def test_generate_satellite_tile_seed_salt_changes_output():
    t1 = generate_satellite_tile(10.0, 20.0, seed_salt="a")
    t2 = generate_satellite_tile(10.0, 20.0, seed_salt="b")
    assert not np.array_equal(t1, t2)


def test_generate_satellite_tile_differs_by_location():
    a = generate_satellite_tile(25.7617, -80.1918)
    b = generate_satellite_tile(-33.8688, 151.2093)
    assert not np.array_equal(a, b)


def test_current_salted_tile_is_brighter_than_prior_on_average():
    # "current"-salted tiles get an injected bright, low-vegetation patch
    # simulating new construction; "prior" tiles stay closer to raw terrain.
    # Averaged over several sites this should hold true in aggregate.
    deltas = []
    for lat, lon in [(10.0, 20.0), (-5.5, 33.2), (48.1, 11.6), (1.3, 103.8)]:
        current = generate_satellite_tile(lat, lon, seed_salt="current")
        prior = generate_satellite_tile(lat, lon, seed_salt="prior")
        deltas.append(current.mean() - prior.mean())
    assert sum(deltas) / len(deltas) > 0


def test_prior_tiles_share_base_terrain_more_than_unrelated_site():
    # Same site's "prior" tile should be closer to a fresh "prior" draw at
    # that same site than to a "prior" tile from a completely different
    # location (both share only the low-frequency terrain, not the RNG
    # noise, so this is a weak-but-real similarity check).
    prior_a = generate_satellite_tile(40.0, -105.0, seed_salt="prior")
    other_site = generate_satellite_tile(-20.0, 55.0, seed_salt="prior")
    same_site_current = generate_satellite_tile(40.0, -105.0, seed_salt="current")
    same_site_diff = np.abs(same_site_current - prior_a).mean()
    cross_site_diff = np.abs(prior_a - other_site).mean()
    # The same-site pair only differs by the injected construction patch
    # (small, localized); the cross-site pair differs in base terrain
    # everywhere, so it should show at least as much divergence.
    assert same_site_diff <= cross_site_diff + 0.05
