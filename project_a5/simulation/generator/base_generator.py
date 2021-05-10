import numpy as np
from project_a5.simulation.particle import BaseParticle


class BaseParticleGenerator:
    """
    Base particle generator class.

    This class is intended as a base class for particle generators.
    Each particle type must create their own generator class if the particle
    has new parameters other than the ones sampled here.

    A derived class must implement:
        __init__
        generate(num, random_seed)

    Attributes
    ----------
    direction : float or None
        The direction of the particle in radians if it is provided.
        If None: the generator will sample directions uniformly in [0, 2pi).
    e_min : float
        Minimum energy for the generated particles, must be 0 <= e_min <= e_max
    e_max : float
        Maximum energy for the generated particles, must be 0 <= e_min <= e_max
    gamma : float
        The index of the power law function from which the energies are
        sampled. Must be >= 1, powerlaw is defined as E^(-gamma)
    name : str
        The name of the particle generator.
    particle_class : BaseParticle class, optional
            A particle class derived from BaseParticle.
            This defines the type of particle that will be created.
    """

    def __init__(
        self,
        e_min,
        e_max,
        gamma,
        direction=None,
        particle_class=BaseParticle,
        name='BaseParticleGenerator',
        particle_args={}
    ):
        """Initialize base particle generator.

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
        name : str, optional
            Optional name of the particle generator.
        """
        assert gamma >= 1, 'gamma must be >= 1 (E^(-gamma))'
        assert 0 <= e_min <= e_max, 'energy limits must be 0 <= e_min <= e_max'
        msg = 'direction must be None or direction in [0, 2pi)'
        assert direction is None or 0 <= direction < 2 * np.pi, msg

        self.name = name
        self.e_min = e_min
        self.e_max = e_max
        self.gamma = gamma
        self.direction = direction
        self.particle_class = particle_class
        self.particle_args = particle_args

        # create a counter that keeps track of number of created particles
        self.particle_id_counter = 0

    def generate(self, num, random_state):
        """Generate num new particles.

        Parameters
        ----------
        num : int
            The number of particles to generate.
        random_state : np.random.Generator or something with the same API
        """
        # create new particle ids
        particle_ids = self._get_particle_ids(num)

        # sample necessary parameters
        energies = self._generate_energies(
            num=num,
            e_min=self.e_min,
            e_max=self.e_max,
            gamma=self.gamma,
            random_state=random_state
        )
        directions = self._generate_directions(
            num=num,
            direction=self.direction,
            random_state=random_state,
        )

        # create particles
        particles = []
        for particle_id, energy, direction in zip(particle_ids,
                                                  energies,
                                                  directions):
            particles.append(self.particle_class(
                energy=energy,
                direction=direction,
                particle_id=particle_id,
                name=self.name,
                **self.particle_args,
                ))
        return particles

    def _get_particle_ids(self, num):
        """Generates particle identification numbers

        Parameters
        ----------
        num : int
            The number of particle identification numbers to generate.

        Returns
        -------
        array_like
            The particle identification numbers
        """
        particle_ids = np.arange(self.particle_id_counter,
                                 self.particle_id_counter + num)

        # update particle id counter
        self.particle_id_counter += num
        return particle_ids

    def _generate_directions(self, num, direction, random_state):
        """Sample random directions.

        Parameters
        ----------
        num : int
            Number of random numbers to generate.
        direction : float, array_like, or None
            If None: directions will be sampled uniformly in [0, 2pi).
            If float: directions will all be set to this value
            If array_like: the directions will be set to these values which
                means that the length has to match num.
        random_state : None, optional
            A random number service.
        """
        # sample uniformly in [0, 2pi)
        if direction is None:
            directions = random_state.uniform(high=2*np.pi, size=num)

        # Set all directions to the same value
        elif isinstance(direction, (int, float)):
            directions = np.repeat(direction, num)

        # Use provided directions
        else:
            if len(direction) != num:
                msg = 'Provided directions do not have the correct '
                msg += 'length: {} != {}'
                raise ValueError(msg.format(len(direction), num))
            directions = np.array(direction)

        return directions

    def _generate_energies(
        self,
        num,
        e_min,
        e_max,
        gamma,
        random_state,
    ):
        r"""
        Sample num events from a power law with index gamma between e_min and
        e_max by using the analytic inversion method.
        The power law pdf is given by

        .. math::
           \mathrm{pdf}(\gamma) = x^{-\gamma} / \mathrm{norm}

        where norm ensures an area under curve of one. Positive spectral index
        gamma means a falling spectrum.

        Note: When :math:`\gamma=1` the integral is

        .. math::
           \int 1/x \mathrm{d}x = ln(x) + c

        This case is also handled.

        Sampling of power laws over multiple order of magnitude with the
        rejection method is VERY inefficient.

        Parameters
        ----------
        num : int
            Number of random numbers to generate.
        e_min, e_max : float
            Border of the pdf, needed for proper normalization.
        gamma : float
            Power law index.
        random_state : None, optional
            A random number service.

        Returns
        -------
        np.ndarray
            The random numbers sampled from a powerlaw.
        """
        u = random_state.uniform(size=num)

        if gamma == 1:
            return np.exp(u * np.log(e_max / e_min)) * e_min
        else:
            radicant = (
                u
                * (e_max**(1 - gamma) - e_min**(1 - gamma))
                + e_min**(1 - gamma)
            )
            return radicant**(1 / (1 - gamma))
