import numpy as np

from project_a5.simulation.particle import BaseParticle


class TrackParticle(BaseParticle):

    """This class implements a track-like particle.

    Energy depositions are distributed along the track in equal distances
    until the particle either has no more energy left or until it has
    propagated the specified distance.
    The track of the particle is defined by an anchor-point, the direction,
    and the propagation distance before and after the anchor-point:
    `prior_propagation_distance` and `post_propagation_distance`, respectively.

    Attributes
    ----------
    direction : float
        The direction of the particle in radians.
        This is an angle inbetween [0, 2pi).
    energy : float
        The energy of the particle. The energy uses arbitrary units.
    x : float
        The x-coordinate of the track anchor-point in detector units.
    y : TYPE
        The y-coordinate of the track anchor-point in detector units.
    """

    def __init__(self, energy, direction, x, y,
                 particle_id=0,
                 name='TrackParticle',
                 prior_propagation_distance=150.,
                 post_propagation_distance=150.,
                 propagation_step_size=5,
                 constant_de_dx=10.,
                 stochastic_relative_de_dx=(0.001, 0.01),
                 ):
        """Initialize the track particle.

        Parameters
        ----------
        energy : float
            The energy of the particle in arbitrary units. This must be greater
            equal zero.
        direction : float
            The direction of the particle in radians. The direction must be
            within [0, 2pi).
        x : float
            The x-coordinate of the track anchor-point in detector units.
        y : float
            The y-coordinate of the track anchor-point in detector units.
        particle_id : int, optional
            An optional particle identification number.
        name : str, optional
            An optional name of the particle.
        prior_propagation_distance : float, optional
            The propagation distance before the anchor-point.
            Note: the actual track length might be less than this, if the
            particle runs out of energy.
        post_propagation_distance : float, optional
            The propagation distance after the anchor-point.
            Note: the actual track length might be less than this, if the
            particle runs out of energy.
        propagation_step_size : float, optional
            The distance between two energy depositions along the track.
        constant_de_dx : float, optional
            The track particle looses a constant amount of energy
            (independent of the particle energy) per length unit, dE/dx.
            This value defines how big that contribution is.
            The unit is energy / 1 detector unit length.
        stochastic_relative_de_dx : (float, float), optional
            In addition to constant energy depositions along the track, the
            particle may also loose energy in stochastic processes.
            The provided values define the lower (min) and upper (max) bound
            of the fractional energy loss per unit step.
            The fractional stochastic energy losses are sampled uniformly
            log-space according to:
                E_loss = E * exp(Uniform(log(min*s), log(max**s)))
            where s is the `propagation_step_size`.
        """

        # call init from base class: this will assign the energy and direction
        super().__init__(
            energy=energy,
            direction=direction,
            particle_id=particle_id,
            name=name,
        )

        # assign values to object
        self.x = x
        self.y = y
        self.prior_propagation_distance = prior_propagation_distance
        self.post_propagation_distance = post_propagation_distance
        self.propagation_step_size = propagation_step_size
        self.constant_de_dx = constant_de_dx
        self.stochastic_relative_de_dx = np.array(stochastic_relative_de_dx)

        # compute constant energy deposition per propagation step
        self.energy_const = self.constant_de_dx * self.propagation_step_size

        # compute range of relative stochastic energy losses
        self.energy_rel = np.log(np.clip(
            self.stochastic_relative_de_dx * self.propagation_step_size,
            0., 1.))

    def propagate(self, random_state, **kwargs):
        """Propagate the particle.

        This method propagates the particle and creates energy losses.
        The energy losses can be passed on to a Detector instance to generate
        an event.

        Parameters
        ----------
        random_state : RNG object
            The random number generator to use.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array_like
            The list of energy losses. Each energy loss consists of a tuple of
            [x-position, y-position, deposited Energy, direction].
            Shape: [num_losses, 4]
        """

        # get direction vector
        dx = np.cos(self.direction)
        dy = np.sin(self.direction)

        # calculate vertex of track
        vertex_x = self.x - dx * self.prior_propagation_distance
        vertex_y = self.y - dy * self.prior_propagation_distance

        # compute total propagation distance
        propagation_distance = (self.prior_propagation_distance +
                                self.post_propagation_distance)

        # create an empty list to store the energy depositions
        energy_depositions = []

        # now move along the particle track with a constant step size
        # and create energy depositions
        energy = self.energy
        distance = 0
        while energy > 0 and distance < propagation_distance:

            # move along the track to the next energy deposition
            distance += self.propagation_step_size
            pos_x = vertex_x + distance * dx
            pos_y = vertex_y + distance * dy

            # sample stochastic energy loss
            stochastic_fraction = np.exp(
                random_state.uniform(self.energy_rel[0], self.energy_rel[1]))

            # calculate energy deposition
            energy_loss = self.energy_const + energy * stochastic_fraction

            # update remaining energy
            energy -= energy_loss

            # correct energy depostion if we went below zero
            if energy < 0:
                energy_loss += energy
                energy = 0

            # append energy deposition
            energy_depositions.append(
                [pos_x, pos_y, energy_loss, self.direction])

        return energy_depositions
