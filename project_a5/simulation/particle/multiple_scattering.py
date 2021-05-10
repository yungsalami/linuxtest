import numpy as np

from project_a5.simulation.particle import BaseParticle
from project_a5.simulation.detector.angle_dist import DeltaAngleDistribution


class MultipleScatteringParticle(BaseParticle):

    """Class implements a particle with multiple scattering and absorption.

    The particle can scatter and be absorbed according the defined
    scattering and absorption lengths. The total propagation length is drawn
    from an exponential with the absorption length as decay parameter.
    The next scattering point is also drawn form an exponential with the
    scattering length as decay parameter.
    The scattering is fully elastic, e.g. the amount of energy deposited at
    a scattering point is zero. At the point of absorption the particle
    deposits all of its energy `energy`.

    Note: this is how Photons are propageted in many particle experiments.

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
                 name='MultipleScatteringParticle',
                 scattering_length=10.,
                 absorption_length=150.,
                 angle_distribution=DeltaAngleDistribution(),
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
        scattering_length : float, optional
            This parameter controls the distance to the next scattering point.
            The distance to the next scattering point is sampled from an
            exponential with `scattering_length` as the decay parameter.
        absorption_length : float, optional
            This parameter controls the total propagation distance until the
            particle is absorbed by the medium. The propagation distance is
            sampled from an exponential with `absorption_length` as the decay
            parameter.
        angle_distribution : DeltaAnglePDF, optional
            The delta angle distribution to use to sample new delta direction
            vectors. Default distribution is: `DeltaAngleDistribution`.
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
        self.scattering_length = scattering_length
        self.absorption_length = absorption_length
        self.angle_distribution = angle_distribution

    def propagate(self, random_state, **kwargs):
        """Propagate the particle.

        This method propagates the particle and creates energy losses.
        The energy losses can be passed on to a Detector instance to generate
        an event.

        The first energy deposition should consist of the vertex of the
        particle, e.g. it should be:
            (self.x, self.y, 0., self.direction)
        The last energy deposition should be the point of absorption and the
        deposited energy should be equal to the particle's energy.

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

        # ---------
        # Exercise:
        # ---------
        #   MC Multiple Scattering (exercises/mc_multiple_scattering.py)
        #
        # ---------------------------------------------------------------------
        # --- Replace Code Here
        # ---------------------------------------------------------------------

        # dummy solution to pass unit tests (this solution is not correct!)
        # this is just a dummy energy deposition list with the starting vertex
        # and one interaction point at a distance of 10 units in x-direction
        energy_depositions = [(self.x, self.y, 0., self.direction),
                              (10., 0., self.energy, 0.)]

        return energy_depositions
