"""This file tests the BaseParticle class.
"""
import pytest
import numpy as np

from project_a5.simulation.particle import BaseParticle


def test_particle_creation_works():
    """Test if BaseParticle instatiation works
    """
    energy = 10.
    direction = 0.2

    base_particle = BaseParticle(energy, direction)

    assert base_particle.name == 'BaseParticle'
    assert base_particle.particle_id == 0
    assert base_particle.energy == energy
    assert base_particle.direction == direction


def test_wrong_energy_init():
    """Initializing the particle with a negative energy should raise expection
    """
    with pytest.raises(AssertionError):
        BaseParticle(-1, 0.4)


def test_direction_out_of_bounds_init():
    """Initializing the particle with a direction which is out of bounds
    should work and the direction should automatically be transformed to the
    correct bounds
    """
    particle = BaseParticle(1, -0.4)
    assert particle.direction == 2*np.pi - 0.4

    particle = BaseParticle(1, 2 * np.pi)
    assert particle.direction == 0.


def test_abstract_propagate_method():
    """The propagate method of the BaseParticle class is an abstract methods.
    A derived class should implement it. Attempting to execute the propagate
    method of the base particle should raise a NotImplementedError.
    """
    base_particle = BaseParticle(10., 0.2)
    with pytest.raises(NotImplementedError, match=r".*abstract method.*"):
        base_particle.propagate()


def test_equality():
    """Test particle equality method
    """
    base_particle1 = BaseParticle(10., 0.2)
    base_particle2 = BaseParticle(10., 0.2)
    base_particle3 = BaseParticle(10., 0.7)

    assert base_particle1 == base_particle2
    assert not base_particle1 == base_particle3
    assert base_particle1 != base_particle3
