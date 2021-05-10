import numpy as np
from scipy.optimize import minimize

from project_a5.simulation.particle import TrackParticle, CascadeParticle


class CascadeDirectionLikelihoodReco():

    """Cascade Direction Likelihood

    This class defines a cascade likelhood reconstruction.
    The likelihood assumes a single energy deposition
    (without saturation and noise). It is defined over the measured charge at
    each pixel.
    However, it does not constrain the total measured charge and is therefore
    not suitable to reconstruct the energy of the cascade.

    Note: this likelihood is an oversimplification and it has numerous
    drawbacks. The main ones being:
        - Noise is not handled
        - Saturation is not handled
        - It is assumed that the PDF does not change within the area of the
          pixel, e.g. this is only accurate if the pixel area is sufficiently
          small
        - Corner clippers are not handled correctly and will lead to bias
    """

    def __init__(self, detector):
        """Set up Cascade Likelihood with a given detector instance.

        Parameters
        ----------
        detector : Detector object
            A detector object for which the cascade likelihood will be set up.
        """
        self.detector = detector

    def negative_log_likelihood(self, cascade, pixels):
        """Compute the negative log likelihood for a given set of pixels.

        The likelihood is defined as a product over the likelihood of each
        pixel weighted by the charge of the pixel.

        Parameters
        ----------
        cascade : array_like
            The direction cascade hypothesis which is a tuple consisting of:
            (x, y, direction)
        pixels : array_like
            The pixel response of the event for which to compute the
            likelihood.
            Shape: [num_pixels_x, num_pixels_y]

        Returns
        -------
        float
            The negative log likelihood.
        """

        direction, x, y = cascade

        # make sure direction is in bounds
        if direction < 0.:
            direction = (direction + (1+int(-direction))*2*np.pi) % (2*np.pi)
        elif direction > 2*np.pi:
            direction = direction % (2*np.pi)

        # compute distances to pixels from cascade
        # Shape: [num_pixels_x, 1]
        dx = np.expand_dims(self.detector.x - x, axis=-1)

        # Shape: [1, num_pixels_y]
        dy = np.expand_dims(self.detector.y - y, axis=0)

        # Shape: [num_pixels_x, num_pixels_y]
        distance = np.sqrt(dx**2 + dy**2)
        direction_pixel = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)

        # calculate signed delta angle
        delta_dir = direction_pixel - direction
        delta_dir[delta_dir < -np.pi] += 2*np.pi
        delta_dir[delta_dir >= np.pi] -= 2*np.pi

        # compute likelihood for each pixel
        # Shape: [num_pixels_x, num_pixels_y]
        likelihood = self.detector.likelihood(delta_dir, distance)

        # positive constant for more stability
        eps = 1e-7

        # Using the negative log likelihood reduces product to sum
        # Shape: ()
        neg_log_likelihood = np.sum(-pixels*np.log(likelihood + eps))

        return neg_log_likelihood

    def reconstruct(self, event, x0=None, **kwargs):
        """Reconstruct an event assuming the cascade likelihood.

        Parameters
        ----------
        event : Event
            The event which we want to reconstruct.
        x0 : tuple, optional
            Our intitial guess for the cascade.
        **kwargs
            Optional keyword arguments that will be passed on to the minimizer.

        Returns
        -------
        CascadeParticle
            The reconstructed cascade particle.
        OptimizeResult
            The optimization result of the scipy minimizer.
        """

        if x0 is None:
            # use weighted pixel average as initial guess
            x0 = (
                0.,
                np.average(self.detector.pixel_x, weights=event.pixels),
                np.average(self.detector.pixel_y, weights=event.pixels),
            )

        # define function to minimize
        def fun(x0):
            return self.negative_log_likelihood(x0, event.pixels)

        optimizer_result = minimize(fun, x0=x0, **kwargs)

        # create reconstructed cascade
        cascade = CascadeParticle(
            energy=0.,
            direction=optimizer_result.x[0],
            x=optimizer_result.x[1],
            y=optimizer_result.x[2])

        return cascade, optimizer_result
