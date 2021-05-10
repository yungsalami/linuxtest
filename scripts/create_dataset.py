import click

import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from sys import exit

from project_a5.io import EventWriter
from project_a5.simulation import Detector, VertexParticleGenerator
from project_a5.simulation.particle import TrackParticle
from project_a5.reconstruction.preprocessing import FeatureGenerator


@click.command()
@click.argument("group-name", type=str)
@click.option(
    "--output-directory",
    "-d",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for generated data and plots.",
)
@click.option(
    "--seed", "-s", default=1337, type=int, help="Random number generator seed"
)
@click.option("--num-events", "-n", default=10000, help="Number of events to simulate.")
def main(group_name, output_directory, seed, num_events):
    """Dataset Creation

    This script will create a dataset consisting of particles from
    the following sources:

        - Diffuse cascade background
        - Diffuse track background
        - Track point source

    The created events will randomly contain an event from one of these
    classes, or it will be a pure noise event.
    """
    # ---
    # Set up output
    # ---
    if output_directory is not None:
        output_directory = Path(output_directory) / group_name
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        print("Skipping creation of dataset to save resources because no <output-directory> was given.")
        exit(1)

    t_start = time.perf_counter()

    # ------------------------------
    # create random number generator
    # ------------------------------
    rg = np.random.default_rng(seed)

    # ---------------
    # set up detector
    # ---------------
    detector = Detector(
        noise_level=10,
        trigger_kernel_shape=[4, 4],
        resolution=50,
        max_saturation_level=300,
    )
    FG = FeatureGenerator(detector)

    diff_cascade_generator = VertexParticleGenerator(
        1e3, 1e6, 1.5, name="DiffuseCascades"
    )
    diff_track_generator = VertexParticleGenerator(
        1e3, 1e6, 1.5, particle_class=TrackParticle, name="DiffuseTracks"
    )
    ps_track_generator = VertexParticleGenerator(
        1e3,
        1e6,
        1.5,
        direction=4.28,
        particle_class=TrackParticle,
        name="TrackPointSource",
    )
    generators = [
        diff_cascade_generator,
        diff_track_generator,
        ps_track_generator,
        None,
    ]

    # ---------------
    # simulate Events
    # ---------------
    print("Simulating Events")

    events = []
    d = dict()

    output_path_events = output_directory / "events.h5"

    with EventWriter(output_path_events, detector, mode="w") as writer:
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
            event = detector.generate_event(particle, rg)
            events.append(event)
            writer.write_event(event)

            features = FG.analyse(event)
            d[event.event_id] = features

    output_path_features = output_directory / "features.h5"
    df = pd.DataFrame.from_dict(d, orient="index")
    df.to_hdf(output_path_features, "events")

    t_diff = time.perf_counter() - t_start
    print(f"Simulating {num_events} events took {t_diff:3.3}s")
    print(f"Find events here: {output_path_events}")

    # ----------------------------
    # Investigate simulated Events
    # ----------------------------

    # get events that triggered the detector
    trigger_events = [e for e in events if e.passed_trigger]

    # calculate fraction of events that passed the detector trigger
    percentage = float(len(trigger_events)) / num_events * 100
    print("A fraction of {:4.2f} % triggered the detector".format(percentage))

    # calculate fractions of particles belonging to each class
    print("Event Fractions of triggered events:")
    for generator in generators:
        if generator is None:
            name = "Noise"
            n_events = np.sum([1 for e in trigger_events if e.particles == []])
        else:
            name = generator.name
            n_events = np.sum(
                [
                    1
                    for e in trigger_events
                    if (e.particles and e.particles[0].name == name)
                ]
            )

        percentage = float(n_events) / len(trigger_events) * 100
        print("\t{}: {:4.2f} %".format(name, percentage))


if __name__ == "__main__":
    main()
