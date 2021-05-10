import numpy as np


class BaseParticle:
    """This defines an abstract base class for Particles.

    Every particle must at least have an associated energy and direction.
    Derived classes may add as many additional particle attributes as desired.

    Attributes
    ----------
    direction : float
        The direction of the particle in radians.
        This is an angle inbetween [0, 2pi).
    energy : float
        The energy of the particle. The energy uses arbitrary units.
    """

    def __init__(self, energy, direction, particle_id=0, name='BaseParticle'):
        """Initialize the particle.

        Parameters
        ----------
        energy : float
            The energy of the particle in arbitrary units. This must be greater
            equal zero.
        direction : float
            The direction of the particle in radians. The direction must be
            within [0, 2pi).
        particle_id : int, optional
            An optional particle identification number.
        name : str, optional
            An optional name of the particle.
        """

        # transform direction in bounds [0, 2pi)
        direction = self.transform_direction_in_bounds(direction)

        # sanity checks
        # throws AssertionError with message of the string if not true
        assert energy >= 0, 'Energy must be positive'
        assert 0 <= direction < 2 * np.pi, 'Direction must be inside [0, 2π)'

        # assign values to object
        self.name = name
        self.energy = energy
        self.direction = direction
        self.particle_id = particle_id

    def transform_direction_in_bounds(self, direction):
        """Transforms a direction angle to the correct bounds [0, 2pi)

        Parameters
        ----------
        direction : float
            The direction to transform

        Returns
        -------
        float
            The transformed direction
        """
        if direction < 0.:
            direction = (direction + (1+int(-direction))*2*np.pi) % (2*np.pi)
        elif direction >= 2*np.pi:
            direction = direction % (2*np.pi)
        return direction

    def propagate(self):
        """Propagate the particle.

        This method propagates the particle and creates energy losses.
        The energy losses can be passed on to a Detector instance to generate
        an event.

        Returns
        -------
        array_like
            The list of energy losses. Each energy loss consists of a tuple of
            [x-position, y-position, deposited Energy, direction].
            Shape: [num_losses, 4]
        """
        raise NotImplementedError('This is an abstract method that needs to '
                                  'be implemented by derived class.')

    def __eq__(self, other):
        """Check for equality of two particles.

        Ignores a particle's name and identification number, e.g. a particle
        is equal to another particle, if its physical parameters are the same.

        Parameters
        ----------
        other : BaseParticle object
            The other particle to compare against.

        Returns
        -------
        bool
            True, if particles are equal
        """
        for key in self.__dict__.keys():

            # ignore particle_id and name as these are not physical quantities
            if key not in ['particle_id', 'name']:
                if self.__dict__[key] != other.__dict__[key]:
                    return False

        return True
