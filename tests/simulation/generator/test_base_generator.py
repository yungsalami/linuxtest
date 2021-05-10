"""This file tests the BaseParticleGenerator class.

Note: Results are only tested for correct shape and reproducibility. Their
actual values are not tested for correctness!
"""
import pytest
import numpy as np

from project_a5.simulation.particle import BaseParticle
from project_a5.simulation.generator import BaseParticleGenerator


def test_base_particle_generator_creation():
    e_min = 10
    e_max = 1000
    gamma = 2
    generator = BaseParticleGenerator(e_min, e_max, gamma)

    assert generator.name == 'BaseParticleGenerator'
    assert generator.e_min == e_min
    assert generator.e_max == e_max
    assert generator.particle_class is BaseParticle
    assert generator.direction is None


def test_wrong_direction_init():
    """Initializing the generator with a wrong direction should raise expection
    """
    e_min = 10
    e_max = 1000
    gamma = 2

    with pytest.raises(AssertionError):
        BaseParticleGenerator(e_min, e_max, gamma, -1)

    with pytest.raises(AssertionError):
        BaseParticleGenerator(e_min, e_max, gamma, 2*np.pi)


def test_wrong_energy_limits_init():
    """Initializing the generator with wrong energy limits should raise
    an expection.
    """
    # minimum energy limit is negative
    with pytest.raises(AssertionError):
        BaseParticleGenerator(-1, 2, 2)

    # maximum energy limit is negative
    with pytest.raises(AssertionError):
        BaseParticleGenerator(1, -2, 2)

    # lower limit is higher than upper energy limit
    with pytest.raises(AssertionError):
        BaseParticleGenerator(3, 2, 2)


def test_generate_energies_shape():
    """Test if energy generation produces correctly shaped results
    """
    e_min = 10
    e_max = 1000
    gamma = 1
    num = 12

    generator = BaseParticleGenerator(e_min, e_max, gamma)

    rg = np.random.default_rng()
    energies = generator._generate_energies(num, e_min, e_max, gamma, rg)

    assert energies.shape == (num,)


def test_generate_energies_reproducibility():
    """Test if energy generation produces reproducible results
    """
    e_min = 10
    e_max = 1000
    gamma = 2
    num = 12
    seed = 42

    generator = BaseParticleGenerator(e_min, e_max, gamma)

    random_state = np.random.default_rng(seed)
    energies1 = generator._generate_energies(num, e_min, e_max, gamma,
                                             random_state=random_state)

    # running it again without reseting seed should produce different results
    energies_diff = generator._generate_energies(num, e_min, e_max, gamma,
                                                 random_state=random_state)
    assert not np.allclose(energies1, energies_diff)

    # now let us pass a custom random service and see if we can reproduce
    random_state = np.random.default_rng(seed)
    energies3 = generator._generate_energies(num, e_min, e_max, gamma,
                                             random_state=random_state)
    assert np.allclose(energies1, energies3)


def test_generate_directions_error():
    """Generate directions can be executed with given directions.
    In this case the shape must match the number of directions to create.
    An exception should be raised if these do not match.
    """
    e_min = 10
    e_max = 1000
    gamma = 1
    num = 12
    num_dir = 7

    generator = BaseParticleGenerator(e_min, e_max, gamma)

    msg = 'Provided directions do not have the correct length: {} != {}'

    rg = np.random.default_rng(0)
    # minimum energy limit is negative
    with pytest.raises(ValueError, match=msg.format(num_dir, num)):
        generator._generate_directions(
            num, direction=np.zeros(num_dir), random_state=rg
        )


def test_generate_directions_shape():
    """Test if direction generation produces correctly shaped results
    """
    e_min = 10
    e_max = 1000
    gamma = 1
    num = 12

    rg = np.random.default_rng()
    generator = BaseParticleGenerator(e_min, e_max, gamma)

    directions = generator._generate_directions(num, direction=None, random_state=rg)
    assert directions.shape == (num,)

    directions = generator._generate_directions(num, direction=0.2, random_state=rg)
    assert directions.shape == (num,)
    assert (directions == 0.2).all()

    directions_true = np.random.uniform(high=6, size=num)
    directions = generator._generate_directions(num, directions_true, rg)
    assert np.allclose(directions, directions_true)


def test_generate_directions_reproducibility():
    """Test if direction generation produces reproducible results
    """
    e_min = 10
    e_max = 1000
    gamma = 2
    num = 12
    direction = None
    seed = 42

    generator = BaseParticleGenerator(e_min, e_max, gamma)

    rg = np.random.default_rng(seed)
    # set numpy random number seed
    directions1 = generator._generate_directions(num, direction, rg)
    direction_diff = generator._generate_directions(num, direction, rg)
    assert not np.allclose(directions1, direction_diff)

    directions2 = generator._generate_directions(
        num, direction=direction, random_state=np.random.default_rng(seed)
    )
    assert np.allclose(directions1, directions2)


def test_generate_same_direction_and_energy():
    """Test particle generation with constant energy and direction.
    """
    num = 4
    e_min = 10
    e_max = 10
    gamma = 2
    direction = .4

    random_state = np.random.default_rng(42)

    generator = BaseParticleGenerator(e_min, e_max, gamma, direction,
                                      name='BackgroundEvents')

    particles = generator.generate(num, random_state)

    assert len(particles) == num

    for i, particle in enumerate(particles):
        assert particle.name == 'BackgroundEvents'
        assert particle.particle_id == i
        assert particle.energy == e_min
        assert particle.direction == direction
        assert isinstance(particle, BaseParticle)


def test_generate():
    """Test particle generation
    """
    num = 4
    e_min = 10
    e_max = 1000
    gamma = 2
    direction = None

    generator = BaseParticleGenerator(e_min, e_max, gamma, direction,
                                      name='PointSource')

    random_state = np.random.default_rng(42)
    particles = generator.generate(num, random_state)

    assert len(particles) == num

    for i, particle in enumerate(particles):
        assert particle.name == 'PointSource'
        assert particle.particle_id == i
        assert particle.energy >= e_min
        assert particle.energy <= e_max
        assert particle.direction >= 0
        assert particle.direction < 2*np.pi
        assert isinstance(particle, BaseParticle)


def test_particle_ids():
    """Test particle identification counter.
    """
    num_1 = 4
    num_2 = 7
    e_min = 10
    e_max = 1000
    gamma = 2
    direction = None

    generator = BaseParticleGenerator(e_min, e_max, gamma, direction)

    # create num_1 particles
    random_state = np.random.default_rng(42)
    particles = generator.generate(num_1, random_state)

    for i, particle in enumerate(particles):
        assert particle.particle_id == i

    # now create num_2 more particles
    particles = generator.generate(num_2, random_state)

    # The particle ids should have not been reset
    for i, particle in enumerate(particles):
        assert particle.particle_id == i + num_1
