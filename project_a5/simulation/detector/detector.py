import numpy as np
from scipy.stats import halfnorm
from scipy.stats import poisson
from scipy.signal import convolve2d

from .angle_dist import DeltaAngleDistribution
from .event import Event
from ..particle import BaseParticle


class Detector:
    """Detector Class.

    This class sets up and defines a detector which consists of a rectangular
    patch of pixels. The detector class is also responsible for simulating
    the detector response given a number of particles.

    Simulation of detector response follows these steps:

        1.) Simulate charge received at each pixel:
            For each particle:
                For each energy deposition:
                    Sample distance from vertex via a Half-Gaussian Disribution
                    Sample delta direction from the `angle_distribution`.
                    The new direction is computed via:
                        new_dir = old_dir + delta_dit
                    Resulting values above 2*pi and below 0 are mapped back to
                        [0, 2pi).
                    Calculate positions of sampled points
                    Apply detector efficiency and randomly discard points
                    Histogram sampled points in pixels
                    Accumulate the histograms (this is the pixel charge)

        2.) Add noise simulation
        3.) Add saturation to pixels
        4.) Apply detector trigger

    Attributes
    ----------
    noise_level : TYPE
        Description
    num_pixels_x : TYPE
        Description
    num_pixels_y : TYPE
        Description
    resolution : TYPE
        Description
    x_extend : TYPE
        Description
    y_extend : TYPE
        Description
    """

    def __init__(
        self,
        x_extend=100,
        y_extend=100,
        num_pixels_x=64,
        num_pixels_y=64,
        resolution=10.0,
        noise_level=0.1,
        detection_probability=0.5,
        relative_saturation_threshold=0.8,
        max_saturation_level=1000,
        trigger_kernel_shape=None,
        trigger_false_positive_rate=0.01,
        num_splits=1000,
        angle_distribution=DeltaAngleDistribution(),
    ):
        """Creates and sets up detector.

        Parameters
        ----------
        x_extend : float, optional
            The extend how far out the detector stretches along the x-axis.
            The detector lower left corner will be placed at (0, 0).
        y_extend : float, optional
            The extend how far out the detector stretches along the y-axis.
            The detector lower left corner will be placed at (0, 0).
        num_pixels_x : int, optional
            The number of pixels along the x-axis.
        num_pixels_y : int, optional
            The number of pixels along the y-axis.
        resolution : float, optional
            This parameter defines the detector resolution.
            This is the capability to spatially resolve individual energy
            depositions. The distance PDF for the charge distribution of
            an energy deposition is a halfnorm with the `resolution` as a
            scale factor.
        noise_level : float or array_like, optional
            This describes the noise level for the pixels.
            If a float is provided, the same noise level is applied to all
            pixels. If an array is given, the noise_level(i, j) corresponds
            to the ith pixel along the x-axis and jth pixel along the y-axis.
        detection_probability : float, optional
            This is the probability of detecting a photon that reaches the
            pixel detector.
        relative_saturation_threshold : float, optional
            This defines the relative fraction of the maximum charge at which
            saturation begins. This must be a value between [0, 1].
            Pixels begin to saturate when they collect a charge above
            `relative_saturation_threshold` * `max_saturation_level`.
        max_saturation_level : float, optional
            The maximum charge a pixel can measure. The pixels start to
            saturate below this value (see `relative_saturation_threshold`)
            and asymptotically approach `max_saturation_level`.
        trigger_kernel_shape : list of int, optional
            The shape of the convolution kernel for the trigger.
            A kernel shape of [1, 1] means that every pixel will be tested
            individually, whereas a kernel shape == event shape will compute
            and test the total event charge.
            Defaults to event shape.
        trigger_false_positive_rate : float, optional
            The false positive rate to achieve for the event trigger.
            Note: this is only approximative due to discreteness of poisson
            and binomial distribution.
        num_splits : int, optional
            This parameter defines the granularity of the energy deposition
            simulation. An energy deposition is divided into `num_splits` parts
            which are distributed according to the distance and angular pdf.
            A higher number of splits will improve the simulation granularity,
            but will also rquire more computational time.
        angle_distribution : DeltaAnglePDF, optional
            The delta angle distribution to use to sample new delta direction
            vectors. Default distribution is: `DeltaAngleDistribution`.
        """
        self.x_extend = x_extend
        self.y_extend = y_extend

        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y
        self.event_shape = (self.num_pixels_x, self.num_pixels_y)

        # create bin edges of pixels
        self.x_bins = np.linspace(0, self.x_extend, self.num_pixels_x + 1)
        self.y_bins = np.linspace(0, self.y_extend, self.num_pixels_y + 1)

        # create pixel center positions
        self.x = self.x_bins[:-1] + np.diff(self.x_bins) / 2.
        self.y = self.y_bins[:-1] + np.diff(self.y_bins) / 2.
        self.pixel_x, self.pixel_y = np.meshgrid(self.x, self.y)

        self.resolution = resolution
        self.noise_level = noise_level
        self.detection_probability = detection_probability
        self.relative_saturation_threshold = relative_saturation_threshold
        self.max_saturation_level = max_saturation_level
        self.num_splits = num_splits

        # define distribution for delta angles
        self.angle_distribution = angle_distribution

        # define distance distribution which is a half Gaussian disribution
        self.distance_distribution = halfnorm(scale=resolution)

        # calculate distance cutoff over which we will not simulate an energy
        # deposition, because we can't expect any light from it.
        # We will set that distance at which we expect to see less than
        # 0.1 percent of the light
        distance_from_center_to_corner = 0.5 * np.sqrt(
            self.x_extend**2 + self.y_extend**2
        )
        self.distance_cutoff = (
            self.distance_distribution.ppf(0.999)
            + distance_from_center_to_corner
        )
        self.distance_cutoff_square = self.distance_cutoff**2

        # calculate trigger threshold
        if trigger_kernel_shape is None:
            self.trigger_kernel_shape = self.event_shape
        else:
            self.trigger_kernel_shape = trigger_kernel_shape
        self.trigger_false_positive_rate = trigger_false_positive_rate
        self.trigger_threshhold = self.compute_trigger_threshold(
            self.trigger_kernel_shape, self.trigger_false_positive_rate)

        # create a counter for created events
        self.event_counter = 0

    def likelihood(self, delta_dir, distance):
        """Likelihood for a given delta direction and distance.

        Parameters
        ----------
        delta_dir : float or array_like
            The angle between the direction of the energy deposition and the
            point at which to evaluate the angular distribution pdf.
        distance : float or array_like
            The distance between the energy deposition and the point at which
            to evalute the distance pdf.

        Returns
        -------
        float or array_like
            The likelihood value for a given delta direction and distance.
        """
        distance_pdf = self.distance_distribution.pdf(distance)
        angle_pdf = self.angle_distribution.pdf(delta_dir)
        return distance_pdf * angle_pdf

    def compute_trigger_threshold(self, kernel_shape, false_positive_rate):
        """Compute the trigger threshold.

        This computes the the trigger treshold for a given kernel shape in
        order to obtain a false positive rate of `false_positive_rate`.

        Parameters
        ----------
        kernel_shape : list of int
            The shape of the convolution kernel.
            A kernel shape of [1, 1] means that every pixel will be tested
            individually, whereas a kernel shape == event shape will compute
            the total event charge.
        false_positive_rate : float
            The false positive rate to achieve.
            Note: this is only approximative due to discreteness of poisson
            and binomial distribution.
        """

        # make sure we have a constant noise level for all pixels
        if not np.isscalar(self.noise_level):
            raise NotImplementedError('Currently only supports constant '
                                      'noise level over all pixels.')

        """
        The probability of having at least one positive outcome in n trials
        should be at most alpha = `false_positive_rate`.
            -> Binomial distribution with probability p and n = `num_tests`.
        The probability of having at least 1 positive outcome in n trials
        with a probability of p is:
        p_binomial = 1 - CDF(0)
                   = 1 - (1 - p)^n

        We can now set the false positive rate (alpha), e.g. the
        probability of having at least 1 positive outcome:

            alpha := p_binomial = 1 - CDF(0) = 1 - (1 - p)^n
            <=> p = 1 - (1 - alpha)^(1/n)

        The probability p is given by a poisson distribution for each patch
        of convolved pixels with an expected charge of
            lambda = num_pixels * noise_level

        The probability p is defined as the probability that the summed charge
        in that patch of convolved pixels is greater than a certain treshhold:

            p = 1 - Poisson(lambda).CDF(treshhold)

        Therefore, the threshold is:

            <=> Poisson(lambda).CDF(treshhold) = (1 - p)
            <=> treshhold = Poisson(lambda).ppf(1 - p)
            <=> treshhold = Poisson(lambda).ppf(1 - (1 - (1 - alpha)^(1/n)))
            <=> treshhold = Poisson(lambda).ppf((1 - alpha)^(1/n))

        """

        # compute number of pixels accumulated in one convolution patch
        num_pixels = np.prod(kernel_shape)

        # compute expected charge lambda for a patch of convolved pixels
        expected_charge = self.noise_level * num_pixels

        # compute number of trials (parameter n of the binomial distribution)
        result_shape = convolve2d(np.zeros(self.event_shape),
                                  np.zeros(kernel_shape),
                                  mode='valid').shape
        num_tests = np.prod(result_shape)

        # compute threshold for desired false positive rate
        threshold = poisson(expected_charge).ppf(
            (1 - false_positive_rate)**(1./num_tests))

        return threshold

    def apply_trigger(self, pixels):
        """Apply event trigger.

        Parameters
        ----------
        pixels : array_like
            The pixel response of the event for which to apply the trigger.
            Shape: [num_pixels_x, num_pixels_y]

        Returns
        -------
        bool
            Returns True if the event passed the trigger,
            False if not.
        """

        # Implement shortcut for special cases for efficiency
        if self.trigger_kernel_shape == self.event_shape:
            result = np.sum(pixels)
        elif self.trigger_kernel_shape == [1, 1]:
            result = pixels
        else:
            kernel = np.ones(self.trigger_kernel_shape)
            result = convolve2d(pixels, kernel, mode='valid')

        return np.any(result > self.trigger_threshhold)

    def add_noise_simulation(self, pixels, random_state):
        """Apply noise simulation to pixels.

        Parameters
        ----------
        pixels : array_like
            The pixels for which to apply the noise simulation.
            Shape: [num_pixels_x, num_pixels_y]
        random_state : RNG object
            The random number generator to use.

        Returns
        -------
        array_like
            Returns the modified pixels including the noise simulation.
            Shape: [num_pixels_x, num_pixels_y]
        """

        # add random noise from poisson distribution
        noise = random_state.poisson(
            lam=self.noise_level, size=self.event_shape
        )
        noise_event = pixels + noise
        return noise_event

    def add_pixel_saturation(self, pixels):
        """Apply pixel saturation.

        The pixels saturate at a certain charge threshold.
        The following saturation curve is applied:
            f(x) =  x, for x <= x_s
                    s(1 - exp(s(1-f) + f*s*ln(f) -x) / (f*2), for x > x_s
        with:
            s: the maximum (asymptotically) reachable charge level
            x_s = s*(1-f): the x value at which saturation starts

        Parameters
        ----------
        pixels : array_like
            The pixels for which to apply the noise simulation.
            Shape: [num_pixels_x, num_pixels_y]

        Returns
        -------
        array_like
            Returns the modified pixels including the saturation.
            Shape: [num_pixels_x, num_pixels_y]
        """

        # compute threshold at which saturation begins
        x_s = self.max_saturation_level * self.relative_saturation_threshold

        # temporary helper variables
        s = self.max_saturation_level
        f = 1 - self.relative_saturation_threshold

        # compute saturation
        saturation_pixels = np.where(
            pixels > x_s,
            s * (1 - np.exp((s*(1-f) + f*s*np.log(f) - pixels)/(f*s))),
            pixels
        )
        return saturation_pixels

    def simulate_particle(self, particle, random_state):
        """Simulate a particle's charge depositions in the detector.

        Parameters
        ----------
        particle : particle
            A particle object that implements a propagate(**kwargs) method
            which returns a list of energy losses (x, y, energy, direction).
        random_state : RNG object
            The random number generator to use.

        Returns
        -------
        array_like
            The charge in each pixel.
            Shape: [num_pixels_x, num_pixels_y]

        Raises
        ------
        ValueError
            Description
        """

        # propagate the particle and obtain the energy depositions
        depositions = np.asarray(particle.propagate(random_state=random_state))

        # allow a particle to not create any energy depositions
        if depositions is None or len(depositions) == 0:
            return np.zeros(shape=self.event_shape)

        if depositions.shape[1:] != (4,):
            msg = 'Unexpected shape for energy depositions of particle `{}`: '
            msg += '{} does not match expected dimension [?, 4].'
            raise ValueError(msg.format(particle.name, depositions.shape))

        # discard any depositions which are too far away from detector
        dist_to_center_sq = (self.x_extend / 2 - depositions[:, 0])**2 + (
                             self.y_extend / 2 - depositions[:, 1])**2
        mask = dist_to_center_sq < self.distance_cutoff_square
        depositions = depositions[mask]

        # split up energy depositions
        # An energy of 1 is chosen to correspond to 1 photon, we can not split
        # any smaller than this. For efficiency, we also don't want to split
        # a large energy deposition into all of its parts. Instead we will
        # choose to split sufficiently large energy depositions in num_splits
        # parts
        deposition_list = []

        # First: choose all energy depositions above an energy of num_splits.
        # We will treat these the same and split them into parts of num_splits.
        mask = depositions[:, 2] >= self.num_splits
        deposition_parts = depositions[mask]

        # adjust energy
        deposition_parts[:, 2] /= self.num_splits

        # create num_splits many of them and append them to the deposition list
        deposition_list.append(np.tile(
            deposition_parts, reps=[self.num_splits, 1]
        ))

        # now we need to go through low energy depositions and split them up
        # into their smallest parts
        for deposition in depositions[~mask]:

            splits = max(1, int(deposition[2]))

            # adjust energy
            deposition[2] /= splits

            # split up the deposition into even parts and append to list
            deposition_list.append(np.tile(
                np.expand_dims(deposition, axis=0),
                reps=[splits, 1],
            ))

        # convert list into numpy array and count how many we have
        depositions = np.concatenate(deposition_list, axis=0)
        num_depositions = len(depositions)

        # sample delta directions for the parts
        delta_dir = self.angle_distribution.rvs(
            size=num_depositions, random_state=random_state
        )

        # sample distances for the parts
        distances = self.distance_distribution.rvs(
            size=num_depositions, random_state=random_state
        )

        x = depositions[:, 0]
        y = depositions[:, 1]
        phi = depositions[:, 3]

        # calculate new sample points
        sample_x = x + distances * np.cos(phi + delta_dir)
        sample_y = y + distances * np.sin(phi + delta_dir)
        sample_weight = depositions[:, 2]

        # sample charge at pixel through poisson distribution
        sample_weight = random_state.poisson(lam=sample_weight)

        # The detector only has limited resolution and will not be able to
        # detect every incoming photon. Randomly set charges to zero
        indices = random_state.choice(
            np.arange(sample_weight.size),
            replace=False,
            size=int(sample_weight.size * (1 - self.detection_probability))
        )
        sample_weight[indices] = 0

        # histogram into bins
        hist, _, _ = np.histogram2d(
            x=sample_x, y=sample_y,
            bins=(self.x_bins, self.y_bins),
            weights=sample_weight,
        )

        return hist

    def generate_event(self, particles, random_state):
        """Generate an event for a given list of particles

        Parameters
        ----------
        particles : Particle or list of Particles
            The event will be created based upon the energy depositions
            created from these particles
        random_state : RNG object
            The random number generator to use.

        Returns
        -------
        array_like
            Returns the generated event.
            Shape: [num_pixels_x, num_pixels_y]
        """

        # create empty pixels
        pixels = np.zeros(shape=self.event_shape)

        # allow a single particle to be passed in
        if isinstance(particles, BaseParticle):
            particles = [particles]

        # simulate collected charge in pixels for each of the paticles
        for particle in particles:
            pixels += self.simulate_particle(particle, random_state)

        # add noise
        pixels = self.add_noise_simulation(pixels, random_state)

        # add saturation
        pixels = self.add_pixel_saturation(pixels)

        # add cutoffs/sensitivity?

        # add trigger information
        passed_trigger = self.apply_trigger(pixels)

        # create event
        event = Event(
            pixels=pixels,
            particles=particles,
            passed_trigger=passed_trigger,
            event_id=self.event_counter,
        )

        # increment event counter
        self.event_counter += 1

        return event
