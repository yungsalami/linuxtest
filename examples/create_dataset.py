import numpy as np
import time
from tqdm import tqdm
import tempfile

from project_a5.io import EventWriter
from project_a5.simulation import Detector, VertexParticleGenerator
from project_a5.simulation.particle import TrackParticle


def main():
    """Dataset Creation Example

    This is an example how to create a dataset consisting of particles from
    the following sources:

        - Diffuse cascade background
        - Diffuse track background
        - Track point source

    The created events will randomly contain an event from one of these
    classes, or it will be a pure noise event.
    """

    # ------------------------------
    # create random number generator
    # ------------------------------
    rg = np.random.default_rng(1337)

    # ---------------
    # set up detector
    # ---------------
    detector = Detector()

    # --------------------------
    # set up particle generators
    # --------------------------
    num_events = 1000

    diff_cascade_generator = VertexParticleGenerator(
        1e3, 1e6, 2, name='DiffuseCascades'
    )
    diff_track_generator = VertexParticleGenerator(
        1e3, 1e6, 2, particle_class=TrackParticle, name='DiffuseTracks'
    )
    ps_track_generator = VertexParticleGenerator(
        1e3, 1e6, 2,
        direction=4.28,
        particle_class=TrackParticle,
        name='TrackPointSource'
    )
    generators = [diff_cascade_generator, diff_track_generator,
                  ps_track_generator, None]

    # ---------------
    # simulate Events
    # ---------------
    print('Simulating Events')
    t_start = time.perf_counter()
    events = []
    for event_id in tqdm(range(num_events)):

        # randomly draw the generator for this event
        generator = rg.choice(generators)

        # generate 1 particle
        if generator is None:
            # this is pure noise, so we will add an empty list
            particle = []
        else:
            particle = generator.generate(1, random_state=rg)

        # simulate and append event
        events.append(detector.generate_event(particle, rg))

    t_diff = time.perf_counter() - t_start
    print('Simulating {} events took: {:3.3}s'.format(num_events, t_diff))

    # ----------------------------
    # Investigate simulated Events
    # ----------------------------
    # get events that triggered the detector
    trigger_events = [e for e in events if e.passed_trigger]

    # calculate fraction of events that passed the detector trigger
    percentage = float(len(trigger_events)) / num_events * 100
    print('A fraction of {:4.2f} % triggered the detector'.format(percentage))

    # calculate fractions of particles belonging to each class
    print('Event Fractions of triggered events:')
    for generator in generators:
        if generator is None:
            name = 'Noise'
            n_events = np.sum([1 for e in trigger_events if e.particles == []])
        else:
            name = generator.name
            n_events = np.sum(
                [1 for e in trigger_events if
                 (e.particles and e.particles[0].name == name)]
            )

        percentage = float(n_events) / len(trigger_events) * 100
        print('\t{}: {:4.2f} %'.format(name, percentage))

    # --------------------
    # Write events to file
    # --------------------
    t_start = time.perf_counter()

    # we will create and use a temporary file
    with tempfile.NamedTemporaryFile(prefix='test_io', suffix='.hdf5') as f:

        # write the events into the file
        with EventWriter(f.name, detector, mode='w') as writer:
            for event in events:
                writer.write_event(event)

    t_diff = time.perf_counter() - t_start
    print('Wrting {} events took: {:3.3}s'.format(num_events, t_diff))


if __name__ == '__main__':
    main()
