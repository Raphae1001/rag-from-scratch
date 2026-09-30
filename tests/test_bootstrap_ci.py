import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from bootstrap_ci import bootstrap_ci  # noqa: E402


def test_bootstrap_ci_mean_matches_sample_mean():
    values = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    mean, lo, hi = bootstrap_ci(values, n_resamples=2000, rng=rng)
    assert mean == values.mean()
    assert lo <= mean <= hi


def test_bootstrap_ci_zero_variance_gives_a_point_interval():
    values = np.array([1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    mean, lo, hi = bootstrap_ci(values, n_resamples=500, rng=rng)
    assert mean == lo == hi == 1.0


def test_bootstrap_ci_is_reproducible_with_the_same_seed():
    values = np.array([0.2, 0.8, 0.5, 0.9, 0.1, 0.6])
    a = bootstrap_ci(values, n_resamples=1000, rng=np.random.default_rng(42))
    b = bootstrap_ci(values, n_resamples=1000, rng=np.random.default_rng(42))
    assert a == b


def test_bootstrap_ci_narrows_around_mean_with_more_data():
    rng = np.random.default_rng(1)
    small = rng.binomial(1, 0.7, size=10).astype(float)
    large = np.tile(small, 20)  # même moyenne, 20x plus de "requêtes"
    _, lo_small, hi_small = bootstrap_ci(small, n_resamples=3000, rng=np.random.default_rng(2))
    _, lo_large, hi_large = bootstrap_ci(large, n_resamples=3000, rng=np.random.default_rng(2))
    assert (hi_large - lo_large) < (hi_small - lo_small)
