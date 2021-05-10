"""This file tests the VertexParticleGenerator class.

Note: Results are only tested for correct shape and reproducibility. Their
actual values are not tested for correctness!
"""
import pytest
from numpy.random import default_rng

from project_a5.simulation.particle import CascadeParticle
from project_a5.simulation.generator import VertexParticleGenerator


def test_vertex_particle_generator_creation():
    """Test the vertex particle generator instantiation
    """
    e_min = 10
    e_max = 1000
    gamma = 2
    x_max = 23.
    generator = VertexParticleGenerator(e_min, e_max, gamma, x_max=x_max)

    assert generator.name == 'VertexParticleGenerator'
    assert generator.e_min == e_min
    assert generator.e_max == e_max
    assert generator.particle_class == CascadeParticle
    assert generator.direction is None
    assert generator.x_min == 0.
    assert generator.x_max == x_max
    assert generator.y_min == 0.
    assert generator.y_max == 100.


def test_vertex_particle_generator_wrong_vertex_limits():
    """Test if exceptions are thrown if wrong vertex limits are provided.
    """
    e_min = 10
    e_max = 1000
    gamma = 2
    x_max = 23.

    with pytest.raises(AssertionError):
        VertexParticleGenerator(
            e_min, e_max, gamma,
            x_max=x_max, x_min=x_max*2
        )

    with pytest.raises(AssertionError):
        VertexParticleGenerator(
            e_min, e_max, gamma,
            y_max=x_max, y_min=x_max*2
        )

    # setting lower and upper limite to the same value should work
    VertexParticleGenerator(
        e_min, e_max, gamma,
        x_max=4, x_min=4, y_min=6, y_max=6
    )


def test_generate_same_parameters():
    """Test particle generation with constant parameters.
    """
    num = 4
    e_min = 10
    e_max = 10
    x_lim = 12.
    y_lim = 1337.
    gamma = 2
    direction = .4
    rg = default_rng()

    generator = VertexParticleGenerator(
        e_min, e_max, gamma, direction,
        x_max=x_lim, x_min=x_lim, y_min=y_lim, y_max=y_lim,
        name='CascadeSource')

    particles = generator.generate(num, rg)

    assert len(particles) == num

    for i, particle in enumerate(particles):
        assert particle.name == 'CascadeSource'
        assert particle.particle_id == i
        assert particle.energy == e_min
        assert particle.direction == direction
        assert particle.x == x_lim
        assert particle.y == y_lim
        assert isinstance(particle, CascadeParticle)


def test_generate():
    """Test particle generation reproducibility
    """
    num = 4
    e_min = 10
    e_max = 1000
    gamma = 2
    direction = None
    seed = 1337

    generator = VertexParticleGenerator(e_min, e_max, gamma, direction)

    particles = generator.generate(num, default_rng(seed))
    particles_diff = generator.generate(num, default_rng(seed + 1))
    particles_same = generator.generate(num, default_rng(seed))

    assert len(particles) == num

    for particle, particle_diff, particle_same in zip(particles,
                                                      particles_diff,
                                                      particles_same):
        assert particle == particle_same
        assert not particle == particle_diff
        assert particle != particle_diff


def test_particle_ids():
    """Test particle identification counter.
    """
    num_1 = 4
    num_2 = 7
    e_min = 10
    e_max = 1000
    gamma = 2
    direction = None
    generator = VertexParticleGenerator(e_min, e_max, gamma, direction)

    # create num_1 particles
    particles = generator.generate(num_1, default_rng(0))

    for i, particle in enumerate(particles):
        assert particle.particle_id == i

    # now create num_2 more particles
    particles = generator.generate(num_2, default_rng(0))

    # The particle ids should have not been reset
    for i, particle in enumerate(particles):
        assert particle.particle_id == i + num_1
