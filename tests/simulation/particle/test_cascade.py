"""This file tests the CascadeParticle class.
"""
import pytest
import numpy as np

from project_a5.simulation.particle import CascadeParticle


def test_particle_creation_works():
    """Test if CascadeParticle instatiation works
    """
    energy = 10.
    direction = 0.2
    x = 12.4
    y = 534.

    cascade = CascadeParticle(energy, direction, x, y)

    assert cascade.name == 'CascadeParticle'
    assert cascade.particle_id == 0
    assert cascade.energy == energy
    assert cascade.direction == direction
    assert cascade.x == x
    assert cascade.y == y


def test_wrong_energy_init():
    """Initializing the particle with a negative energy should raise expection
    """
    with pytest.raises(AssertionError):
        CascadeParticle(-1, 0.4, 23, 1)


def test_direction_out_of_bounds_init():
    """Initializing the particle with a direction which is out of bounds
    should work and the direction should automatically be transformed to the
    correct bounds
    """
    particle = CascadeParticle(1, -0.4, 23, 1)
    assert particle.direction == 2*np.pi - 0.4

    particle = CascadeParticle(1, 2*np.pi, 23, 1)
    assert particle.direction == 0.


def test_propagate_method():
    """The propagate method of the CascadeParticle class is an abstract methods.
    A derived class should implement it. Attempting to execute the propagate
    method of the base particle should raise a NotImplementedError.
    """
    energy = 10.
    direction = 0.2
    x = 12.4
    y = 534.

    true_energy_deposition = np.array([[x, y, energy, direction]])

    cascade = CascadeParticle(energy, direction, x, y)
    energy_depositions = cascade.propagate()

    assert np.allclose(true_energy_deposition, energy_depositions)
