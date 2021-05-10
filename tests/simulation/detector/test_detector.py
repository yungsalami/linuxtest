"""This file tests the Detector class.

Note: Results are only tested for correct shape and reproducibility. Their
actual values are not tested for correctness!
"""
import numpy as np

from project_a5.simulation.particle import CascadeParticle
from project_a5.simulation.detector import Detector


def test_detector_creation():
    """Test the creation of the angular direction distribution.
    """
    y_extend = 72.
    num_pixels_x = 32
    detector = Detector(y_extend=y_extend, num_pixels_x=num_pixels_x)

    assert detector.x_extend == 100.
    assert detector.y_extend == y_extend
    assert detector.num_pixels_x == num_pixels_x
    assert detector.num_pixels_y == 64
    assert detector.event_shape == (num_pixels_x, 64)
    assert detector.resolution == 10.
    assert detector.noise_level == 0.1


def test_likelihood():
    """Test Likelihood function.
    """
    detector = Detector()

    lh_max = detector.likelihood(delta_dir=0., distance=0.)

    assert detector.likelihood(delta_dir=0.2, distance=0.) < lh_max
    assert detector.likelihood(delta_dir=-0.2, distance=0.) < lh_max
    assert detector.likelihood(delta_dir=0., distance=10.) < lh_max


def test_noise_simulation():
    """Test noise simulation.
    """
    detector = Detector()

    rg = np.random.default_rng()
    empty_event = np.zeros(detector.event_shape)

    noise_event = detector.add_noise_simulation(empty_event, rg)

    assert noise_event.shape == detector.event_shape
    assert np.sum(noise_event) > np.sum(empty_event)


def test_event_generation():
    """Test a full event simulation with cascade particles.
    """

    # create a low energy event in lower left part of detector
    cascade_1 = CascadeParticle(energy=10., direction=0.2, x=5., y=5.)

    # and a higher nergy event in upper right part of detector
    cascade_2 = CascadeParticle(energy=10000., direction=0.2, x=70., y=70.)

    # a cascade very far out that should not be simulated or produce any light
    cascade_3 = CascadeParticle(energy=10000., direction=0.2, x=-20., y=-20.)

    # create detector
    detector = Detector()
    detector_no_noise = Detector(noise_level=0.)

    # generate empty event
    rg = np.random.default_rng(0)
    event = detector_no_noise.generate_event([], rg)
    assert event.pixels.shape == detector.event_shape
    assert np.sum(event.pixels) == 0.
    assert event.event_id == 0
    assert event.name == 'Event'
    assert event.particles == []

    # generate event with cascade very far out
    event = detector_no_noise.generate_event([cascade_3], rg)
    assert event.pixels.shape == detector.event_shape
    assert np.sum(event.pixels) == 0.
    assert event.event_id == 1
    assert event.name == 'Event'
    assert event.particles == [cascade_3]

    # generate event from both cascades
    event = detector.generate_event([cascade_1, cascade_2], rg)

    assert event.pixels.shape == detector.event_shape
    assert np.sum(event.pixels) > 0.
    assert event.event_id == 0
    assert event.name == 'Event'
    assert event.particles == [cascade_1, cascade_2]
