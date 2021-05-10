import click

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import time
from pathlib import Path

from project_a5.reconstruction.unfolding import (
    C_thikonov,
    llh_gradient,
    llh_hessian,
    minimize,
)


g = np.array([1, 2])
f = np.array([2, 1])
A = np.array([[0.8, 0.2], [0.2, 0.8]])


def test_C_thikonov():
    return np.all(C_thikonov(2) == [[-1, 1], [1, -1]]) & (np.sum(C_thikonov(10)) == 0)


def test_llh_gradient():
    grad = llh_gradient(A, g, f)
    return np.allclose(grad, [0.22, -0.44], atol=0.01)


def test_llh_hesse():
    hessian = llh_hessian(A, g, f)
    return np.allclose(hessian, [[0.25, 0.27], [0.27, 0.90]], atol=0.01)


@click.command()
@click.argument("group-name")
@click.argument("input-directory", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--output-directory", "-d", default=None)
def main(group_name, input_directory, output_directory):
    # create output directory
    if output_directory is not None:
        output_directory = Path(output_directory) / group_name
        output_directory.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_directory) / group_name / "dataset.h5"

    passed_C_thikonov = test_C_thikonov()
    passed_llh_gradient = test_llh_gradient()
    passed_llh_hesse = test_llh_hesse()
    passed_all = passed_C_thikonov & passed_llh_gradient & passed_llh_hesse

    # ---
    # Exercise
    # ---
    t_start = time.perf_counter()

    df = pd.read_hdf(input_path, "events")

    f_var = "log_energy"
    df[f_var] = np.log10(df["energy"])
    g_var = "log_reco_energy"
    df[g_var] = np.log10(df["reco_energy"])

    # dummy train-test split
    df = df.sample(frac=1, random_state=42)
    split = int(len(df) * 0.5)
    df_train = df[:split]
    df_test = df[split:]

    bins_g = np.linspace(3.5, 5.5, 11)

    bins_f = np.linspace(3.5, 5.5, 15)
    bmid_f = (bins_f[1:] + bins_f[:-1]) / 2.0

    H, _, _ = np.histogram2d(df_train[g_var], df_train[f_var], (bins_g, bins_f))
    A = H / np.sum(H, axis=0)

    g, _ = np.histogram(df_test[g_var], bins_g)
    f, _ = np.histogram(df_test[f_var], bins_f)
    r, _ = np.histogram(df_test[g_var], bins_f)

    delta_t = time.perf_counter() - t_start
    # ---
    # Plots
    # ---
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    gs = plt.GridSpec(2, 3, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1:]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    # ---
    # Migration Matrix
    # ---
    ax = axes[0]
    ax.imshow(A)
    ax.set_title("Migration Matrix")

    # ---
    # Regularisierung
    # ---
    for i, tau in enumerate([1e-6, 1e-3, 1e-1]):
        result = minimize(
            fun_grad=lambda x: llh_gradient(A, g, x, tau),
            fun_hess=lambda x: llh_hessian(A, g, x, tau),
            x0=np.ones(len(f)),
        )

        ax = axes[i + 2]
        ax.errorbar(
            bmid_f,
            f,
            np.sqrt(f),
            np.diff(bins_f) / 2.0,
            linestyle="",
            label="Truth",
            alpha=0.8,
        )
        ax.errorbar(
            bmid_f,
            r,
            np.sqrt(r),
            np.diff(bins_f) / 2.0,
            linestyle="",
            label="Reconstruction",
            alpha=0.4
        )
        ax.errorbar(
            bmid_f,
            result.x,
            np.sqrt(result.hess_inv.diagonal()),
            np.diff(bins_f) / 2.0,
            linestyle="",
            label="Unfolding",
        )
        ax.set_xlabel(f_var)
        ax.set_ylabel("Counts")
        if i == 2:
            ax.legend()
        ax.set_title(f"tau = {tau:.6f}")

    # ---
    # Information Tile
    # ---
    text = "\n".join(
        [
            "Information:",
            "    Exercise: Entfaltung",
            f"    Groupname: {group_name}",
            f"    Runtime: {delta_t:3.4f}s",
            "Tests:",
            f"    Passed thikonov test: {passed_C_thikonov}",
            f"    Passed gradient test: {passed_llh_gradient}",
            f"    Passed hessian test: {passed_llh_hesse}",
        ]
    )

    props = dict(
        boxstyle="square",
        edgecolor="green" if passed_all else "red",
        linewidth=2,
        facecolor="white",
        alpha=0.5,
        pad=1,
    )

    ax = axes[1]
    ax.text(
        0.0, 0.95, text, transform=ax.transAxes, bbox=props, verticalalignment="top"
    )
    ax.axis("off")

    # ------------------
    # Save or show Plots
    # ------------------
    if output_directory is None:
        plt.show()
    else:
        fig.savefig(output_directory / Path(__file__).with_suffix(".pdf").name)


if __name__ == "__main__":
    main()
