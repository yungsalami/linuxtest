import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from project_a5.visualization import EventDisplay
from project_a5.simulation import Detector, VertexParticleGenerator
from project_a5.simulation.particle import TrackParticle, CascadeParticle
from project_a5.reconstruction.likelihood import CascadeDirectionLikelihoodReco
from project_a5.reconstruction.likelihood import ResimulationLikelihood


def make_correlation_plots(particles, recos):
    """Make correlation plots

    Parameters
    ----------
    particles : list of particles
        A list of (MC Truth) particles.
    recos : list of particles
        A list of reconstruced particles.
    """
    labels = ['Vertex X', 'Vertex Y', 'Direction', 'Energy']
    y_true = np.array([(p.x, p.y, p.direction, p.energy) for p in particles]).T
    y_pred = np.array([(p.x, p.y, p.direction, p.energy) for p in recos]).T

    fig, axes = plt.subplots(2, 2)

    axes = axes.flatten()
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    for i, (ax, label) in enumerate(zip(axes, labels)):
        residuals = y_true[i] - y_pred[i]
        axes[i].hexbin(y_true[i], y_pred[i], mincnt=1)
        axes[i].set_xlabel('{} [true]'.format(label))
        axes[i].set_ylabel('{} [pred]'.format(label))
        axes[i].set_title('Mean: {:3.3f} Std-dev: {:3.3f}'.format(
            np.mean(residuals), np.std(residuals)))
    fig.tight_layout()
    plt.show()


def main():
    """Likelihood Reconstruction Example
    """

    # create random number generator
    rg = np.random.default_rng(1337)

    # set up detector
    detector = Detector()

    # -----------------
    # set up likelhoods
    # -----------------
    cascade_llh = CascadeDirectionLikelihoodReco(detector)
    resim_llh = ResimulationLikelihood(detector)

    # --------------------------
    # set up particle generators
    # --------------------------
    num_cascades = 100
    num_tracks = 10

    cascade_generator = VertexParticleGenerator(
        1e3, 1e6, 2, name='CascadeSource'
    )
    track_generator = VertexParticleGenerator(
        1e4, 1e6, 2, particle_class=TrackParticle, name='TrackSource'
    )

    # create particles
    cascades = cascade_generator.generate(num_cascades, random_state=rg)
    tracks = track_generator.generate(num_tracks, random_state=rg)

    # create track and cascade events
    cascade_events = [detector.generate_event(c, rg) for c in cascades]
    track_events = [detector.generate_event(t, rg) for t in tracks]

    # ----------------------
    # Cascade Reconstruction
    # ----------------------
    print('Reconstructing Cascades...')
    cascade_recos = []
    for event_id in tqdm(range(num_cascades)):
        cascade = cascades[event_id]
        cascade_event = cascade_events[event_id]

        # define initial guess (log10_energy, direction, x, y)
        x0 = (
            np.log10(np.sum(cascade_event.pixels)
                     / detector.detection_probability),
            np.pi,  # Seed with constant
            # cascade.direction,  # Seed with MC Truth
            np.average(detector.pixel_x, weights=cascade_event.pixels),
            np.average(detector.pixel_y, weights=cascade_event.pixels),
        )

        cascade_reco, result = cascade_llh.reconstruct(
            cascade_event,
            x0=x0[1:],
            method='Nelder-Mead',
        )

        cascade_recos.append(cascade_reco)

    make_correlation_plots(cascades, cascade_recos)

    # --------------------
    # Track Reconstruction
    # --------------------
    print('Reconstructing Tracks...')
    track_recos = []
    for event_id in tqdm(range(num_tracks)):
        track = tracks[event_id]
        track_event = track_events[event_id]

        # We will cheat and assume we have a fairly decent reconstruction
        # method available that we can use as a seed
        # In addition, we will only reconstruct the direction and leave all
        # other parameters at their true values.
        x0 = (rg.normal(track.direction, scale=0.5),)

        # define x0 transformation function
        def track_x0_to_particles(x0):
            """Helper function to transform hypothesis parameters to particles
            """
            direction, = x0
            return TrackParticle(energy=track.energy,
                                 direction=direction,
                                 x=track.x,
                                 y=track.y)

        track_reco, result = resim_llh.reconstruct(
            event=track_event,
            x0=x0,
            method='Nelder-Mead',
            options=dict(xatol=.01, fatol=100),
            x0_to_particles_function=track_x0_to_particles,
            random_state=rg,
        )
        track_recos.append(track_reco)

    make_correlation_plots(tracks, track_recos)


if __name__ == '__main__':
    main()
