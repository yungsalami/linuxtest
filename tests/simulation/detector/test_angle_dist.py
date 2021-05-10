"""This file tests all specified angle distributions classes.

Note: Results are only tested for correct shape and reproducibility. Their
actual values are not tested for correctness!
"""
import pytest
import numpy as np

from project_a5.simulation.detector.angle_dist import DeltaAngleDistribution
from project_a5.simulation.detector.exp_angle_dist import \
    ExponentialDeltaAngleDistribution

# Define Angle distributions to test
ANGLE_DISTRIBUTIONS = [
    DeltaAngleDistribution,
    ExponentialDeltaAngleDistribution,
]


def test_distribution_creation():
    """Test the creation of the angular direction distribution.
    """
    for Distribution in ANGLE_DISTRIBUTIONS:
        Distribution()


def test_distribution_pdf():
    """Test the pdf.
    """
    for Distribution in ANGLE_DISTRIBUTIONS:
        dist = Distribution()

        num = 12
        delta_dir = np.random.uniform(low=-1, size=num)
        pdf_values = dist.pdf(delta_dir)

        assert len(pdf_values) == num

        # values outside of [-pi, +pi) should be zero
        eps = 1e-15
        assert dist.pdf(-3*np.pi) == 0
        assert dist.pdf(-np.pi - eps) == 0
        assert dist.pdf(np.pi + eps) == 0
        assert dist.pdf(2*np.pi) == 0

        # values at 0. should be the larges
        assert dist.pdf(0.) > dist.pdf(0.1)
        assert dist.pdf(0.) > dist.pdf(-0.1)


def test_distribution_cdf():
    """Test the cdf.
    """
    for Distribution in ANGLE_DISTRIBUTIONS:
        dist = Distribution()

        num = 12
        delta_dir = np.random.uniform(low=-1, size=num)
        cdf_values = dist.cdf(delta_dir)

        assert len(cdf_values) == num

        # values below -pi should be zero, above +pi should be 1
        assert np.allclose(dist.cdf(-3*np.pi), 0)
        assert np.allclose(dist.cdf(-np.pi), 0)
        assert dist.cdf(np.pi) == 1
        assert dist.cdf(2*np.pi) == 1

        # values at 0. should be 0.5
        assert np.allclose(dist.cdf(0.), 0.5)


def test_distribution_ppf():
    """Test ppf method.
    """
    for Distribution in ANGLE_DISTRIBUTIONS:
        dist = Distribution()

        num = 12
        q = np.random.uniform(size=num)
        delta_dir = dist.ppf(q)

        assert len(delta_dir) == num

        # values outside of [0, 1] should raise an exception
        regex = r".*Provided quantiles are out of allowed range *"
        with pytest.raises(ValueError, match=regex):
            dist.ppf(-1.)
        with pytest.raises(ValueError, match=regex):
            dist.ppf(1.00001)
        with pytest.raises(ValueError, match=regex):
            dist.ppf(4.2)

        # values at 0.5 should be 0.
        assert np.allclose(dist.ppf(0.5), 0., rtol=1e-3, atol=1e-3)
        assert np.allclose(dist.ppf(0.), -np.pi, rtol=1e-3, atol=1e-3)
        assert np.allclose(dist.ppf(1.0), np.pi, rtol=1e-3, atol=1e-3)


def test_distribution_rvs():
    """Test rvs method to sample points form the angular distribution.
    """
    for Distribution in ANGLE_DISTRIBUTIONS:
        random_state = np.random.default_rng(42)
        dist = Distribution()

        num = 12
        delta_dir = dist.rvs(random_state, num)

        eps = 1e-15
        assert len(delta_dir) == num
        assert np.all(delta_dir >= -np.pi - eps)
        assert np.all(delta_dir <= np.pi + eps)
