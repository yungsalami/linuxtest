import click

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

from project_a5.io import read_particles
import pickle


def reconstruct_direction(features):
    # This is a dummy reconstruction.
    # Do not change.
    return (
        features["direction"]
        + np.random.default_rng(1337).normal(size=len(features)) * 0.2
    )


def reconstruct_energy(features, path):
    # Do not change.
    with open(path, "rb") as f:
        model = pickle.load(f)

    y = model.predict(features)

    return y


def reconstruct_particle_id(features):
    # We skip particle ID reconstruction for now.
    # Do not change.
    return np.nan


@click.command()
@click.argument("group-name", type=str)
@click.argument("input-directory", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "--output-directory",
    "-d",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option("--direction", is_flag=True, default=False)
@click.option("--energy", is_flag=True, default=False)
@click.option("--particle-id", is_flag=True, default=False)
def main(
    group_name,
    input_directory,
    output_directory,
    direction,
    energy,
    particle_id,
):

    # set up input/output directories and paths
    if output_directory is not None:
        output_directory = Path(output_directory) / group_name
        output_directory.mkdir(parents=True, exist_ok=True)

    input_directory = Path(input_directory) / group_name
    energy_regressor_path = input_directory / "energy_regressor.pkl"
    output_path = output_directory / "dataset.h5"

    # load data
    features = pd.read_hdf(input_directory / "features.h5", "events")
    particles = read_particles(input_directory / "events.h5")
    particles = particles.set_index("event_id")

    features.dropna(inplace=True)
    particles = particles.loc[features.index.intersection(particles.index)]

    if direction:
        # For now, use a dummy reconstruction.
        print("Reconstructing Direction")
        reco_direction = reconstruct_direction(particles)

    if energy:
        print("Reconstructing Energy")
        reco_energy = reconstruct_energy(features, energy_regressor_path)
    else:
        reco_energy = np.nan

    if particle_id:
        print("Reconstructing Particle ID")
        reco_particle_id = reconstruct_particle_id(features)
    else:
        reco_particle_id = np.nan

    features["reco_direction"] = reco_direction
    features["reco_energy"] = reco_energy
    features["reco_particle_id"] = reco_particle_id

    data = features.join(particles, rsuffix="r", how="inner")
    data.to_hdf(output_path, "events")

    # ===
    # Overview Plots
    # ===
    print("Plotting")
    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2, 2, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    ax = axes[0]
    *_, im = ax.hist2d(
        data["direction"], data["reco_direction"], bins=50, norm=LogNorm()
    )
    ax.set_xlabel("Direction")
    ax.set_ylabel("Reconstructed Direction")
    fig.colorbar(im, ax=ax, label="Events")

    ax = axes[1]
    if energy:
        *_, im = ax.hist2d(
            np.log10(data["energy"]),
            np.log10(data["reco_energy"]),
            bins=50,
            norm=LogNorm(),
        )
        ax.set_xlabel("Energy")
        ax.set_ylabel("Reconstructed Energy")
        fig.colorbar(im, ax=ax, label="Events")
    else:
        ax.text(0.1, 0.5, "No Energy Reconstruction")
        ax.axis("off")

    ax = axes[2]
    if particle_id:
        *_, im = ax.hist2d(
            data["particle_id"], data["reco_particle_id"], bins=50, norm=LogNorm()
        )
        ax.set_xlabel("Particle Id")
        ax.set_ylabel("Reconstructed Particle Id")
        fig.colorbar(im, ax=ax, label="Events")
    else:
        ax.text(0.1, 0.5, "No Particle ID Reconstruction")
        ax.axis("off")

    ax = axes[3]
    ax.axis("off")

    # ---
    # Save or Show Plot
    # ---
    if output_directory is None:
        plt.show()
    else:
        fig.savefig(output_directory / Path(__file__).with_suffix(".pdf").name)


if __name__ == "__main__":
    main()
