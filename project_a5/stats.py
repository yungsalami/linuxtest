import numpy as np


def significance(n_on, n_off, alpha):
    """Calculate the significance of an observation.

    Parameters
    ----------
    n_on : int
        Count of events in On-Region.
    n_off : int
        Count of events in Off-Region.
    alpha : float
        Weighting of Off-Region.

    Returns
    -------
    float
    """

    # dummy solution.
    return 0


def test_significance():
    return np.allclose(
        [significance(120, 160, 0.6), significance(150, 320, 0.3)],
        [1.84, 4.38],
        atol=0.01,
    )


def count_in_region(x, xmin, xmax):
    """Count the number of elements between `xmin` and `xmax`.

    Parameters
    ----------
    x : np.array
        Elements to count.
    xmin : float
        Lower bound of region (inclusive).
    xmax : float
        Upper bound of region (exclusive).

    Returns
    -------
    n : count of events in the region
    """
    if xmin > xmax:
        x = (x - xmax) % (2 * np.pi)
        xmin, xmax = xmax, xmin

    n = np.sum((x >= xmin) & (x < xmax))

    return n
