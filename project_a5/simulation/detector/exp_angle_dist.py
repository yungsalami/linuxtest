import numpy as np
from numpy.random import default_rng


class ExponentialDeltaAngleDistribution(object):

    """Helper class for angular distribution PDF

    This is a helper class which defines the angular distribution pdf that
    may be used to describe the difference between the new and and old
    angle:
        new_angle = old_angle + delta_angle
        (plus accounting for 2-pi periodicity)

    The angle PDF used here peaks in forward direction (x=0) and drops off
    exponentially to bigger delta angles. It is defined by:

    f(x) = N * exp(-|x|k), for x in [-pi, pi)
           0, for x < -pi or x >= pi

        k: Parameter that defines how peaked the distribution is
        N: normalization constant
        N = k / (2*[1 - exp(-pi*k)])
    """

    def __init__(self, k=1.):
        """Initialize the distribution.

        Parameters
        ----------
        k : float, optional
            Defines how peaked in forward direction and how sharply the PDF
            alls off towards higher angles.
            The higher k, the sharper the PDF is peaked in forwards direction.
        """
        self.k = k

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

        # ---------
        # Exercise:
        # ---------
        #   Angle PDf | part a) (exercises/angle_pdf.py)
        #
        # define N
        N = self.k/(2 * (1 - np.exp(- np.pi * self.k)))

        values = np.zeros_like(delta_dir)
        if np.isscalar(delta_dir):
            if delta_dir < -np.pi or delta_dir >= np.pi:
                values = 0.
            elif delta_dir == 0.:
                values = N
            elif delta_dir >= -np.pi and delta_dir <= 0:
                values = N * np.exp(delta_dir * self.k)
            elif delta_dir > 0 and delta_dir < np.pi:
                values = N * np.exp(- delta_dir * self.k)
        else:
            mask_neg = np.logical_and(delta_dir >= -np.pi, delta_dir <= 0)
            values[mask_neg] = N * np.exp(delta_dir[mask_neg] * self.k)

            mask_pos = np.logical_and(delta_dir > 0, delta_dir < np.pi)
            values[mask_pos] = N * np.exp(- delta_dir[mask_pos] * self.k)

            values[delta_dir == 0] = N
            # rest can stay 0 as it is produced above

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

        # ---------
        # Exercise:
        # ---------
        #   Angle PDf | part b) (exercises/angle_pdf.py)
        #
        # ---------------------------------------------------------------------
        # --- Replace Code Here
        # ---------------------------------------------------------------------
        values = 0.5*np.ones_like(delta_dir)

        # define N
        N = self.k/(2 * (1 - np.exp(- np.pi * self.k)))

        # values below range have a cdf of zero, over range have cdf of 1
        if np.isscalar(delta_dir):
            if delta_dir <= -np.pi:
                values = 0.
            elif delta_dir >= np.pi:
                values = 1.
            elif delta_dir > -np.pi and delta_dir < 0:
                values = (N/self.k) * (np.exp(delta_dir * self.k) - np.exp(- np.pi * self.k))
            elif delta_dir > 0 and delta_dir < np.pi:
                values = (N/self.k) * (1 - np.exp(-delta_dir * self.k)) + 1/2
            elif delta_dir == 0:
                values = 0.5

        else:
            # masks for the cases
            mask_case1 = np.logical_and(delta_dir > -np.pi, delta_dir < 0)
            mask_case2 = np.logical_and(delta_dir > 0, delta_dir < np.pi)

            values[mask_case1] = (N/self.k) * (np.exp(delta_dir[mask_case1] * self.k) - np.exp(- np.pi * self.k))
            values[mask_case2] = (N/self.k) * (1 - np.exp(-delta_dir[mask_case2] * self.k)) + 1/2

            # special cases
            values[delta_dir == 0] = 0.5
            values[delta_dir >= np.pi] = 1.
            values[delta_dir <= -np.pi] = 0.


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

        # ---------
        # Exercise:
        # ---------
        #   Angle PDf | part c) (exercises/angle_pdf.py)
        #

        # define N
        N = self.k/(2 * (1 - np.exp(- np.pi * self.k)))

        # make value array
        values = np.zeros_like(q)

        # in case q is just one value
        if np.isscalar(q):
            if q == 0.:
                values = -np.pi
            elif q == 0.5:
                values = 0.
            elif q == 1.:
                values = np.pi
            elif q > 0 and q < 0.5:
                values = np.log(self.k * q / N + np.exp(- np.pi * self.k)) /self.k
            elif q > 0.5 and q < 1:
                values = - np.log(1 - (self.k/N) * (q - 1/2))/self.k
             
        else:
            # special cases first
            values[q == 0.0] = -np.pi
            values[q == 0.5] = 0.
            values[q == 1.0] = np.pi

            mask_lower = np.logical_and(q > 0, q < 0.5) # formula for lower percentage
            values[mask_lower] = np.log(self.k * q[mask_lower] / N + np.exp(- np.pi * self.k)) /self.k

            mask_higher = np.logical_and(q > 0.5, q < 1.) # formula for higher percentage
            values[mask_higher] = - np.log(1 - (self.k/N) * (q[mask_higher] - 1/2))/self.k

        return values

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

        # ---------
        # Exercise:
        # ---------
        #   Angle PDf | part d) (exercises/angle_pdf.py)
        #
        # ---------------------------------------------------------------------
        # --- Replace Code Here
        # ---------------------------------------------------------------------

        u = random_state.uniform(0, 1, size) # random uniform numbers
        samples = self.ppf(u) # transform to desired distribution

        return samples
