from project_a5.simulation.particle import BaseParticle


class CascadeParticle(BaseParticle):

    """This class implements a point-like energy deposition (cascade) particle.

    Attributes
    ----------
    direction : float
        The direction of the particle in radians.
        This is an angle inbetween [0, 2pi).
    energy : float
        The energy of the particle. The energy uses arbitrary units.
    x : float
        The x-coordinate of the cascade vertex in detector units.
    y : TYPE
        The y-coordinate of the cascade vertex in detector units.
    """

    def __init__(self, energy, direction, x, y,
                 particle_id=0,
                 name='CascadeParticle'):
        """Initialize the cascade particle.

        Parameters
        ----------
        energy : float
            The energy of the particle in arbitrary units. This must be greater
            equal zero.
        direction : float
            The direction of the particle in radians. The direction must be
            within [0, 2pi).
        x : float
            The x-coordinate of the cascade vertex in detector units.
        y : float
            The y-coordinate of the cascade vertex in detector units.
        particle_id : int, optional
            An optional particle identification number.
        name : str, optional
            An optional name of the particle.
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

    def propagate(self, **kwargs):
        """Propagate the particle.

        This method propagates the particle and creates energy losses.
        The energy losses can be passed on to a Detector instance to generate
        an event.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array_like
            The list of energy losses. Each energy loss consists of a tuple of
            [x-position, y-position, deposited Energy, direction].
            Shape: [num_losses, 4]
        """

        # The cascade particle losses all of its energy in a single energy
        # deposition
        energy_loss = (self.x, self.y, self.energy, self.direction)

        # this method needs to return a list of energy losses, so create a list
        return [energy_loss]
