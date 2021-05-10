import numpy as np

from project_a5.simulation.particle import CascadeParticle
from project_a5.simulation.generator import BaseParticleGenerator


class VertexParticleGenerator(BaseParticleGenerator):

    """Vertex particle generator class.

    This class can generate all particles with parameters
    (x, y, energy, direction). The energies are drawn from a powerlaw
    distribution. The vertex and direction is sampled uniformly in the
    provided boundaries.

    Attributes
    ----------
    e_min : float
        Minimum energy for the generated particles, must be 0 <= e_min <= e_max
    e_max : float
        Maximum energy for the generated particles, must be 0 <= e_min <= e_max
    gamma : float
        The index of the power law function from which the energies are
        sampled.
    direction : float or None
        The direction of the particle in radians if it is provided.
        If None: the generator will sample directions uniformly in [0, 2pi).
    name : str
        The name of the particle generator.
    x_min : float, optional
        The lower bound for the x-position of the particle vertex.
    x_max : float, optional
        The upper bound for the x-position of the particle vertex.
    y_min : float, optional
        The lower bound for the y-position of the particle vertex.
    y_max : float, optional
        The upper bound for the y-position of the particle vertex.
    """

    def __init__(self, e_min, e_max, gamma, direction=None,
                 particle_class=CascadeParticle,
                 x_min=0., x_max=100., y_min=0., y_max=100.,
                 name='VertexParticleGenerator',
                 particle_args={}):
        """Initialize particle generator.

        Parameters
        ----------
        e_min : float
            The minimium particle energy to generate.
            This must be greater equal zero and not greater than e_max.
        e_max : float
            The maximum particle energy to generate.
            This must be greater equal zero and not less than e_min.
        gamma : float
            The index of the power law function from which the energies are
            sampled.
        direction : float or None, optional
            The direction of the particle in radians if it is provided.
            If None: the generator will sample directions uniformly
                     in [0, 2pi).
        particle_class : BaseParticle class, optional
            A particle class derived from BaseParticle.
            This defines the type of particle that will be created.
            The parameters of the particle class must consist of
            (x, y, energy, direction).
        x_min : float, optional
            The lower bound for the x-position of the particle vertex.
        x_max : float, optional
            The upper bound for the x-position of the particle vertex.
        y_min : float, optional
            The lower bound for the y-position of the particle vertex.
        y_max : float, optional
            The upper bound for the y-position of the particle vertex.
        name : str, optional
            Optional name of the particle generator.

        Raises
        ------
        ValueError
            If incorrect boundaries are provided for vertex position.
        """

        # check limits
        msg = 'Incorrect limits for {}: ({}, {}). Upper limit must not be '
        msg += 'smaller than lower limit.'

        assert x_min <= x_max, 'limits must be x_min <= x_max'
        assert y_min <= y_max, 'limits must be y_min <= y_max'

        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max

        # call init of the base class
        super().__init__(
            e_min=e_min, e_max=e_max, gamma=gamma,
            direction=direction,
            particle_class=particle_class,
            name=name,
            particle_args=particle_args
        )

    def generate(self, num, random_state):
        """Generate num new particles.

        Parameters
        ----------
        num : int
            The number of particles to generate.
        random_state : RNG object
            The random number generator to use.
        """

        # create new particle ids
        particle_ids = self._get_particle_ids(num)

        # sample necessary parameters
        energies = self._generate_energies(
            num=num,
            e_min=self.e_min,
            e_max=self.e_max,
            gamma=self.gamma,
            random_state=random_state,
        )
        directions = self._generate_directions(
            num=num,
            direction=self.direction,
            random_state=random_state,
        )
        x_values = random_state.uniform(
            low=self.x_min, high=self.x_max, size=num,
        )
        y_values = random_state.uniform(
            low=self.y_min, high=self.y_max, size=num,
        )

        # create particles
        particles = []
        for x, y, energy, direction, particle_id in zip(x_values,
                                                        y_values,
                                                        energies,
                                                        directions,
                                                        particle_ids):
            particles.append(self.particle_class(
                energy=energy,
                direction=direction,
                x=x,
                y=y,
                particle_id=particle_id,
                name=self.name,
                **self.particle_args,
            ))
        return particles
