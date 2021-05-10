import numpy as np


def test_lcg():
    from project_a5.random import LCG

    rg = LCG(0, a=5, c=7, m=16)

    values = rg.random_raw(size=16)

    assert np.all(values[:3] == np.array([7, 10, 9, ]))
    values.sort()
    assert np.all(values == np.arange(16))


def test_particle_lcg():
    from project_a5.simulation.generator import VertexParticleGenerator
    from project_a5.random import LCG

    gen = VertexParticleGenerator(1, 100, 2.7)
    gen.generate(10, LCG())
