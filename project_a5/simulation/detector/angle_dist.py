import numpy as np


class DeltaAngleDistribution(object):

    """Helper class for angular distribution

    This is a helper class which defines the angular distribution pdf that
    may be used to describe the difference between the new and and old
    angle:
        new_angle = old_angle + delta_angle
        (plus accounting for 2-pi periodicity)

    The angle PDF used here is a parabola that peaks at x=0 (forward direction)
    given by:

    f(x) = 3*(pi^2 - x^2) / (4*pi^3), for x in [-pi, pi)
           0, for x < -pi or x >= pi
    """

    def __init__(self, num_ppf_points=10000):
        """Initialize the distribution."""
        # ppf is computed numerically, we will cache the values here
        self._delta_dir_values = np.linspace(-np.pi, np.pi, num_ppf_points)
        self._cdf_values = self.cdf(self._delta_dir_values)

    def pdf(self, delta_dir):
        """Angular distribution PDF.

        Parameters
        ----------
        delta_dir : float or array_like
            The angle between the direction of the energy deposition and the
            point at which to evaluate the angular distribution PDF.
            Value range: [-pi, pi)

        Returns
        -------
        float or array_like
            The PDF value of the angular distribution.
        """
        values = 3*(np.pi**2 - delta_dir**2) / (4*np.pi**3)

        # values outside of range have a probability of zero
        if np.isscalar(delta_dir):
            if delta_dir < -np.pi or delta_dir >= np.pi:
                values = 0.
        else:
            mask = np.logical_or(delta_dir < -np.pi, delta_dir >= np.pi)
            values[mask] = 0.

        return values

    def cdf(self, delta_dir):
        """Angular distribution CDF.

        Parameters
        ----------
        delta_dir : float or array_like
            The angle between the direction of the energy deposition and the
            point at which to evaluate the angular distribution PDF.
            Value range: [-pi, pi)

        Returns
        -------
        float or array_like
            The PDF value of the angular distribution.
        """
        pi_2 = np.pi**2
        pi_3 = pi_2 * np.pi
        values = (3*pi_2*delta_dir - delta_dir**3 + 2*pi_3) / (4*pi_3)

        # values below range have a cdf of zero, over range have cdf of 1
        if np.isscalar(delta_dir):
            if delta_dir < -np.pi:
                values = 0.
            elif delta_dir >= np.pi:
                values = 1.
        else:
            values[delta_dir < -np.pi] = 0.
            values[delta_dir >= np.pi] = 1.

        return values

    def ppf(self, q):
        """Percent point function (inverse of cdf).

        Parameters
        ----------
        q : float or array_like
            The percentile or quantile (lower tail probability) for which
            to compute the delta direction value.

        Returns
        -------
        float or array_like
            The delta direction value corresponding to the quantile `q`.
        """

        if np.any(q < 0.) or np.any(q > 1.):
            msg = 'Provided quantiles are out of allowed range of [0, 1]: {!r}'
            raise ValueError(msg.format(q))

        # find CDF values that correspond to the specified quantile
        indices = np.searchsorted(self._cdf_values, q, side='left')
        return self._delta_dir_values[indices]

    def rvs(self, random_state, size=None):
        """Sample values from delta_dir PDF.

        Parameters
        ----------
        random_state : TYPE
            Description
        size : None, optional
            Number and shape of delta directions to sample.

        Returns
        -------
        float or array_like
            The sampled delta directions..
        """

        # sample uniform values
        rvs = random_state.uniform(size=size)

        # Use inversion method to transform sampled values to
        # desired distribution
        return self.ppf(rvs)
