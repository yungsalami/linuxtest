import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.random import default_rng

from project_a5.visualization import EventDisplay
from project_a5.simulation import Detector, VertexParticleGenerator


def main():
    rg = default_rng()
    detector = Detector()

    disp = EventDisplay(detector=detector, cmap='inferno')
    disp.add_colorbar()
    disp.fig.show()

    generator = VertexParticleGenerator(1e3, 1e6, 2)

    for particle in generator.generate(50, random_state=rg):
        event = detector.generate_event(particle, random_state=rg)
        disp.set_event(event)
        for i, p in enumerate(event.particles):
            disp.add_particle(p, color=f'C{i}')
        plt.pause(0.5)
        disp.clear_particles()


if __name__ == '__main__':
    main()
