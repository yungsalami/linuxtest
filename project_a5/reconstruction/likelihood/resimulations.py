import numpy as np
from scipy.optimize import minimize


class ResimulationLikelihood():

    """General Likelihood obtained via Resimulations

    This class defines a general likelihood reconstruction for arbirtray event
    hypotheses. The PDF is obtained by avering over n resimulations of the
    same event hypothesis. It does not use an analytic Likelihood.

    Attributes
    ----------
    detector : Detector object
        The detector object for which the likelihood will be set up.
    """

    def __init__(self, detector, method='MSE'):
        """Set up Likelihood with a given detector instance.

        Parameters
        ----------
        detector : Detector object
            A detector object for which the likelihood will be set up.
        method : str, optional
            The likelihood method to use.
            Must be one of: 'MSE', 'Poisson', 'PoissonIncludingTotalCharge'

        """
        self.detector = detector
        self.method = method
        methods = ['MSE', 'Poisson', 'PoissonIncludingTotalCharge']

        if self.method not in methods:
            raise ValueError('Method not understood: {}'.format(method))

    def get_average_pixel_response(self,
                                   particles,
                                   random_state,
                                   n_resimulations):
        """Get the average pixel response via resimulation

        Parameters
        ----------
        particles : Particle or list of Particles
            The particles which make up the event hypothesis.
        random_state : RNG object
            The random number generator to use.
        n_resimulations : int
            Description
        """
        average_pixels = np.zeros(self.detector.event_shape)
        for i in range(n_resimulations):
            average_pixels += self.detector.generate_event(particles,
                                                           random_state).pixels
        average_pixels /= n_resimulations
        return average_pixels

    def negative_log_likelihood(self, particles, pixels, random_state,
                                n_resimulations):
        """Compute the negative log likelihood for a given set of pixels.

        The chosen likelihood depends on the set method during intialisation.
        'Poisson':
            The likelihood is defined as a per pixel Poisson likelihood between
            the actual pixel and the expected pixel response based on the event
            hypothesis.
        'PoissonIncludingTotalCharge':
            Same as Poisson but with an additional term that adds a Poisson
            likelihood for the overall event charge.
        'MSE':
            Per pixel mean squared error between the true and expected pixel
            response based on the event hypothesis.

        Parameters
        ----------
        particles : Particle or List of Particles
            The particles which make up the event hypothesis.
            These particles will be resimulated `n_resimulation` times in
            order to obtain an expected pixel response.
        pixels : array_like
            The pixel response of the event for which to compute the
            likelihood.
            Shape: [num_pixels_x, num_pixels_y]
        random_state : RNG object
            The random number generator to use.
        n_resimulations : int
            The number of resimulations to perform per event hypothesis.
            A higher number will lead to a more accurate reconstruction, but
            will also increase the runtime.
        include_total_charge : bool, optional
            If True, an additional Poisson likelihood term is added for the
            total event charge.

        Returns
        -------
        float
            The negative log likelihood.
        """

        # get expected pixel response
        pixels_pred = self.get_average_pixel_response(
            particles=particles,
            random_state=random_state,
            n_resimulations=n_resimulations,
        )

        if self.method in ['Poisson', 'PoissonIncludingTotalCharge']:
            # positive constant for numerical stability
            eps = 1e-12

            # compute Poisson log likelihood for each pixel
            # Note: this omits the normalization constant which is only
            # dependent on the true pixels and therefore not necessary for the
            # minimization.
            # Shape: (num_pixels_x, num_pixels_y)
            log_likelihood = pixels * np.log(pixels_pred + eps) - pixels_pred

            # compute Poisson log likelhood for total event charge
            if self.method == 'PoissonIncludingTotalCharge':
                event_charge_true = np.sum(pixels)
                event_charge_pred = np.sum(pixels_pred)
                log_likelihood += event_charge_true * np.log(
                    event_charge_pred + eps) - event_charge_pred

            # Using the log likelihood reduces product to sum. Minmize neg llh.
            # Shape: ()
            neg_log_likelihood = np.sum(-log_likelihood)

        elif self.method == 'MSE':
            # Mean Squared Error Loss
            # (e.g. Gauss Assumption with constant uncertainty)
            neg_log_likelihood = np.sum((pixels_pred - pixels)**2)

        # normalize neg likelhood value by total event charge to facilitate
        # tolerance selection for minimizer
        neg_log_likelihood /= np.sum(pixels)

        return neg_log_likelihood

    def reconstruct(self, event, x0, x0_to_particles_function,
                    random_state, n_resimulations=10,
                    **kwargs):
        """Reconstruct an event assuming the cascade directional likelihood.

        Parameters
        ----------
        event : Event
            The event which we want to reconstruct.
        x0 : tuple, optional
            Our intitial guess for the cascade.
        x0_to_particles_function : callable
            A function which transforms the hypothesis parameters x0 to a
            particle or to a list of particles that can be simulated by the
            detector. This enables the flexibility to reconstruct arbitrary
            and possibly compound event hypotheses.

            Example function for a full cascade reconstruction:

                def x0_to_particles_function(x0):
                    energy, direction, x, y = x0
                    cascade = CascadeParticle(
                        energy=energy, direction=direction, x=x, y=y,
                    )
                    return cascade

            Example for a compound event hpyothesis of two particles, where we
            only want to fit for the energy:

                def x0_to_particles_function(x0):
                    energy_cascade, energy_track = x0
                    cascade = CascadeParticle(
                        energy=energy_cascade, direction=direction, x=x, y=y,
                    )
                    track = TrackParticle(
                        energy=energy_track, direction=direction, x=x, y=y,
                    )
                    return [cascade, track]

        random_state : RNG object
            The random number generator to use.
        n_resimulations : int, optional
            The number of resimulations to perform per event hypothesis.
            A higher number will lead to a more accurate reconstruction, but
            will also increase the runtime.
        **kwargs
            Optional keyword arguments that will be passed on to the minimizer.

        Returns
        -------
        Particles or List of Particles
            The reconstructed particles.
        OptimizeResult
            The optimization result of the scipy minimizer.
        """

        # define function to minimize
        def fun(x0):

            # get a list of particles
            particles = x0_to_particles_function(x0)

            return self.negative_log_likelihood(
                particles=particles,
                pixels=event.pixels,
                random_state=random_state,
                n_resimulations=n_resimulations,
            )

        optimizer_result = minimize(fun, x0=x0, **kwargs)
        particles = x0_to_particles_function(optimizer_result.x)

        return particles, optimizer_result
