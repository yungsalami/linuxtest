import click

from pathlib import Path
from sys import exit
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd

from project_a5.stats import significance, test_significance, count_in_region


def is_between(x, x_min, x_max):
    return (x_min <= x) and (x <= x_max)


def regions_overlap(on_region, off_region):
    on_min, on_max = on_region
    off_min, off_max = off_region
    a = is_between(on_min, off_min, off_max)
    b = is_between(on_max, off_min, off_max)
    c = is_between(off_min, on_min, on_max)
    d = is_between(off_min, on_min, on_max)
    return a or b or c or d


def test_regions_overlap():
    assert regions_overlap((0, 1), (0, 1))
    assert not regions_overlap((0, 1), (2, 3))
    assert regions_overlap((0, 1), (1, 2))
    assert regions_overlap((0, 1), (0.1, 1.1))


@click.command()
@click.argument("group-name", type=str)
@click.argument("input-directory", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option(
    "--output-directory", "-d", default=None, help="Folder where plots will be saved to"
)
@click.option("--on-region", default=(0.0, 1.0), help="Bounds for the On-Region.")
@click.option("--off-region", default=(2.0, 3.0), help="Bounds for the Off-Region.")
@click.option("--seed", '-s', default=1337, help="Random Seed")
def main(group_name, input_directory, output_directory, on_region, off_region, seed):
    if not np.diff(on_region) >= 0:
        print(
            "Error: Bitte verwende eine On-Region im Bereich 0 <= lower < upper <= 2π."
        )
        exit(1)
    if not np.diff(off_region) >= 0:
        print(
            "Error: Bitte verwende eine Off-Region im Bereich 0 <= lower < upper <= 2π."
        )
        exit(1)

    # set up output directory
    if output_directory is not None:
        output_directory = Path(output_directory) / group_name
        output_directory.mkdir(parents=True, exist_ok=True)

    events_filename = Path(input_directory) / group_name / "dataset.h5"
    # ===
    # Exercise
    # ===
    t_start = time.perf_counter()

    passed_overlapping_regions = not regions_overlap(on_region, off_region)
    passed_significance = test_significance()

    df = pd.read_hdf(events_filename, "events")
    df = df.sample(frac=0.1, random_state=seed)
    direction = df["reco_direction"].to_numpy()

    n_on = count_in_region(direction, *on_region)
    if n_on == 0:
        print(f"Error: No events found in On-Region: {on_region}.")
        exit(1)

    n_off = count_in_region(direction, *off_region)
    if n_off == 0:
        print(f"Error: No events found in Off-Region: {off_region}.")
        exit(1)

    alpha = (max(on_region) - min(on_region)) / (max(off_region) - min(off_region))

    sigma = significance(n_on, n_off, alpha)

    delta_t = time.perf_counter() - t_start

    # ===
    # Plotting
    # ===
    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2, 2, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0], projection="polar"),
        fig.add_subplot(gs[0, 1], projection="polar"),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    # ---
    # Detection Information
    # ---
    text = "\n".join(
        [
            f"n_on = {n_on}",
            f"n_off = {n_off}",
            f"alpha = {alpha:0.4f}",
            f"significance = {sigma:3.3f}σ",
        ]
    )
    print(text)
    props = dict(
        boxstyle="square",
        edgecolor="gray",
        linewidth=2,
        facecolor="white",
        alpha=0.5,
        pad=1,
    )

    ax = axes[2]
    ax.text(
        0.0, 0.95, text, transform=ax.transAxes, bbox=props, verticalalignment="top"
    )
    ax.axis("off")

    # ---
    # Histogram of Directions
    # ---
    print("Plotting. Takes a bit of time.")

    xbins = 100
    ybins = 1
    n_pts = 101

    ax = axes[0]

    hist, edges = np.histogram(direction, bins=xbins)
    y = np.max(hist)

    ax.fill_between(
        x=np.linspace(*on_region, n_pts), y1=y, color="green", alpha=0.3, label="On",
    )
    ax.fill_between(
        x=np.linspace(*off_region, n_pts), y1=y, color="red", alpha=0.3, label="Off",
    )

    ax.bar(edges[1:], hist, color="w", edgecolor="k", width=0.1)
    ax.set_yticks([])
    ax.set_xlabel("Direction")

    ax.set_ylim(0, y)

    ax.grid(False)

    ax = axes[1]
    hist, edges = np.histogram(direction, bins=xbins)
    color = "magma"
    cmap = cm.get_cmap(color, lut=np.max(hist))
    y = 1
    y2 = y + 0.1

    *_, im = ax.hist2d(
        direction, np.zeros_like(direction), bins=[xbins, ybins], cmap=cmap
    )
    ax.cla()

    ax.bar(edges[1:], y, color=cmap(hist), width=0.1)
    ax.set_yticks([])
    ax.set_xlabel("Direction")

    ax.fill_between(
        x=np.linspace(*on_region, n_pts),
        y1=y,
        y2=y2,
        color="green",
        edgecolor="k",
        alpha=0.3,
    )
    ax.fill_between(
        x=np.linspace(*off_region, n_pts), y1=y, y2=y2, color="red", alpha=0.3,
    )

    ax.set_ylim(0, y2)

    ax.grid(False)

    fig.colorbar(im, ax=ax, label="# Events")
    fig.legend(loc="center", title="Region", ncol=2)

    # ---
    # Information Tile
    # ---
    text = "\n".join(
        [
            "Information:",
            "    Exercise: Testen",
            f"    Groupname: {group_name}",
            f"    Runtime: {delta_t:3.4f}s",
            "Tests:",
            f"    Passed significance test: {passed_significance}",
            f"    Passed overlapping regions test: {passed_overlapping_regions}",
        ]
    )
    passed_all = passed_significance & passed_overlapping_regions

    props = dict(
        boxstyle="square",
        edgecolor="green" if passed_all else "red",
        linewidth=2,
        facecolor="white",
        alpha=0.5,
        pad=1,
    )

    ax = axes[-1]
    ax.text(
        0.0, 0.95, text, transform=ax.transAxes, bbox=props, verticalalignment="top"
    )
    ax.axis("off")

    # ---
    # Save or Show Plot
    # ---
    if output_directory is None:
        plt.show()
    else:
        fig.savefig(output_directory / Path(__file__).with_suffix(".pdf").name)


if __name__ == "__main__":
    test_regions_overlap()
    main()
