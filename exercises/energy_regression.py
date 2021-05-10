import click
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from tqdm import tqdm
import time

from project_a5.simulation import Detector, VertexParticleGenerator
from project_a5.simulation.particle import CascadeParticle, TrackParticle
from project_a5.reconstruction.preprocessing import (
        FeatureGenerator, scale_energy, reverse_energy_scaling
)
from project_a5.reconstruction.machine_learning import (
        define_model, cross_validate_model)


simulation_defaults = {
        'num_cascade': 10000,
        'num_track': 10000,
        'background_level': 10,
        'energy_low': 1e3,
        'energy_high': 1e6,
        'random_seed': 1337,
        'gamma': 2,
        }


def run_test_for_nan_values(X):
    """ Test if Features contain nan values

    Parameters
    ----------
    X: pd.DataFrame
        Analysed events
    """
    return ~X.isnull().values.any()


def run_test_for_proper_settings(
        num_cascade,
        energy_low,
        energy_high,
        gamma,
        background_level,
        random_seed):
    """ Test if simulation settings have not been changed

    Parameters
    ----------
    num_cascade: int
        Number of simulated cascade events
    background level: int
        Detector background level
    energy_low: int
        Lower bound of simulated energies
    energy_high: int
        Upper bound of simulated energies
    random_seed: int
        Random seed for simulation and models
    gamma: int
        Exponent for the energy spectrum
    """
    return ((num_cascade == simulation_defaults['num_cascade'])
            & (background_level == simulation_defaults['background_level'])
            & (energy_low == simulation_defaults['energy_low'])
            & (energy_high == simulation_defaults['energy_high'])
            & (random_seed == simulation_defaults['random_seed'])
            & (gamma == simulation_defaults['gamma'])
            )


def run_test_for_scaling(values, scaling):
    """ Test if the implemented scaling returns the original values
    after the inverse transformation

    Parameters
    ----------
    values: list or np.array
        Particle energies or arbitrary floats
    scaling: str
        Scaling method to apply
    """
    scaled_values = scale_energy(values, scaling)
    reversely_scaled_values = reverse_energy_scaling(scaled_values, scaling)
    return np.allclose(values, reversely_scaled_values)


def simulate_particles(
        detector,
        random_generator,
        num_cascade,
        energy_low,
        energy_high,
        gamma):

    diff_cascade_generator = VertexParticleGenerator(
        energy_low,
        energy_high,
        gamma,
        particle_class=CascadeParticle,
        name='DiffuseCascades'
    )

    diff_track_generator = VertexParticleGenerator(
        energy_low,
        energy_high,
        gamma,
        particle_class=TrackParticle,
        name='DiffuseTracks'
    )

    events = []
    energies = []

    # simulate particles
    cascades = diff_cascade_generator.generate(
            num_cascade,
            random_state=random_generator
            )
    tracks = diff_track_generator.generate(
            num_cascade,
            random_state=random_generator
            )

    particles = cascades + tracks

    print('Simulating Cascade Events')
    for particle in tqdm(particles):
        # propagate and append event
        events.append(detector.generate_event(
            particle,
            random_state=random_generator
        ))
        energies.append(particle.energy)
    return events, energies


def analyze_events(events, FG):
    analysed_events = []
    for event in tqdm(events):
        analysed_events.append(FG.analyse(event))

    return pd.DataFrame(analysed_events)


@click.command()
@click.argument('group_name', type=str)
@click.option('--seed', '-s', default=1337, type=int,
              help='Random number generator seed')
@click.option('--num_cascade', '-nc',
              default=simulation_defaults['num_cascade'], type=int,
              help='Number of cascade events to simulate')
@click.option('--energy_low', '-el',
              default=simulation_defaults['energy_low'], type=int,
              help='Lower bound of simulated energies')
@click.option('--energy_high', '-eh',
              default=simulation_defaults['energy_high'], type=int,
              help='Upper bound of simulated energies')
@click.option('--gamma', '-g',
              default=simulation_defaults['gamma'], type=int,
              help='Exponent for the energy spectrum')
@click.option('--background_level', '-b',
              default=simulation_defaults['background_level'], type=int,
              help='Level of detector background noise')
@click.option('--output_directory', '-d', default=None,
              help='Folder where plots will be saved to')
@click.option('--target_scaling', '-t', default='dummy',
              help='Scaling to apply on energy values')
def main(
        group_name,
        seed,
        num_cascade,
        energy_low,
        energy_high,
        gamma,
        background_level,
        output_directory,
        target_scaling):

    # create output directory
    if output_directory is not None:
        output_directory = Path(output_directory) / group_name
        output_directory.mkdir(parents=True, exist_ok=True)


    # start timer
    t_start = time.perf_counter()

    # set random generator
    rg = np.random.default_rng(seed)

    # ---------------------------
    # Simulate and analyse events
    # ---------------------------
    detector = Detector(
            noise_level=background_level,
            trigger_kernel_shape=[4, 4],
            resolution=50,
            max_saturation_level=300)
    FG = FeatureGenerator(detector)
    events, event_energies = simulate_particles(
            detector,
            rg,
            num_cascade,
            energy_low,
            energy_high,
            gamma)

    print('Analysing events ...')
    X = analyze_events(events, FG)
    y = np.array(event_energies)

    # -------------
    # Test Features
    # -------------
    passed_nan_test = run_test_for_nan_values(X)

    # -------------
    # Test Settings
    # -------------
    passed_settings_test = run_test_for_proper_settings(
            num_cascade,
            energy_low,
            energy_high,
            gamma,
            background_level,
            seed)

    # ------------
    # Test Scaling
    # ------------
    passed_scaling_test = run_test_for_scaling(
            y,
            target_scaling)

    # ------------
    # Scale Target
    # ------------
    y_scaled = scale_energy(y, target_scaling)

    # --------------------------------------
    # Train the model using cross-validation
    # --------------------------------------
    model = define_model(seed)

    print('Performing cross validation ...')
    # obtain energy predictions
    y_cv_pred, y_cv, models = cross_validate_model(X, y_scaled, model, seed)
    # reverse scaling
    y_cv_pred = reverse_energy_scaling(y_cv_pred, target_scaling)
    y_cv = reverse_energy_scaling(y_cv, target_scaling)

    # ---
    # Save model.
    # ---
    if output_directory is not None:
        model.fit(X, y_scaled)
        model.feature_names = X.columns
        model_path = output_directory / "energy_regressor.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # -------------
    # Create Figure
    # -------------
    print('Generating figures ...')
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[1:, 1])
    ax5 = fig.add_subplot(gs[0, 1])

    # ---------------------
    # Draw Confusion Matrix
    # ---------------------
    counts, x_edges, y_edges, img = ax1.hist2d(
            np.log10(y_cv),
            np.log10(y_cv_pred),
            bins=[100, 100],
            norm=LogNorm()
    )
    ax1.set_xlim([np.log10(energy_low), np.log10(energy_high)])
    ax1.set_ylim([np.log10(energy_low), np.log10(energy_high)])
    ax1.set_ylabel('Log10(Predicted Energy)')
    ax1.set_xlabel('Log10(Simulated Energy)')
    ax1.set_title('Regressor Confusion')
    img.set_rasterized(True)
    ax1.figure.colorbar(img, ax=ax1)

    # ------------------------
    # Draw Bias and Resolution
    # ------------------------
    bins = np.logspace(np.log10(1e3), np.log10(1e6), 10)
    df = pd.DataFrame()
    df['bin'] = np.digitize(y_cv, bins)
    df['rel_error'] = (y_cv_pred - y_cv) / y_cv
    df['prediction'] = y_cv_pred
    df['true'] = y_cv

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')
    binned['bias'] = grouped['rel_error'].mean()
    binned['bias_median'] = grouped['rel_error'].median()
    binned['lower_sigma'] = grouped['rel_error'].agg(
            lambda s: np.percentile(s, 15)
    )
    binned['upper_sigma'] = grouped['rel_error'].agg(
            lambda s: np.percentile(s, 85)
    )
    binned['resolution_quantiles'] = (
            (binned.upper_sigma - binned.lower_sigma) / 2
    )
    binned['resolution'] = grouped['rel_error'].std()

    ax2.errorbar(
        binned['center'],
        binned['bias'],
        xerr=0.5 * binned['width'],
        label='Bias',
        linestyle='',
    )
    ax2.errorbar(
        binned['center'],
        binned['resolution'],
        xerr=0.5 * binned['width'],
        label='Resolution',
        linestyle='',
    )
    ax2.set_xlabel('Energy / GeV')
    ax2.set_title('Bias and Resolution')
    ax2.legend()
    ax2.set_xscale('log')

    # -------------------
    # Print some more metrics
    # -------------------
    r2 = r2_score(y_cv_pred, y_cv)
    mse = mean_squared_error(y_cv_pred, y_cv)
    mae = mean_absolute_error(y_cv_pred, y_cv)

    textstr = '\n'.join((
        r'More Metrics:',
        r'      R^2 score: {0:.2f}'.format(r2),
        r'      Mean Squared Error: {0:.2f}'.format(mse),
        r'      Mean Absolute Error: {0:.2f}'.format(mae),
    ))
    props = dict(boxstyle='square', edgecolor='black', linewidth=2,
                 facecolor='white', alpha=0.5, pad=1)

    ax3.text(0.0, 0.95, textstr,
             linespacing=2.,
             verticalalignment='top',
             bbox=props)
    ax3.axis('Off')

    # ------------------------
    # Draw Feature Importances
    # ------------------------
    try:
        feature_importances = np.array([
            tree.feature_importances_
            for forest in models
            for tree in np.array(forest).ravel()])
        order = np.argsort(np.median(feature_importances, axis=0))
        ax4.boxplot(
            feature_importances[:, order],
            vert=False,
            sym='',
            medianprops={'color': 'C0'}
        )
    except:
        print('Could not plot the feature importances.'
              ' Did you replace the dummy regressor ? ')
    ax4.set_xlabel('Feature Importance')
    ax4.set_title('Feature Importances')
    ax4.set_yticklabels(X.columns.values)

    # ----------------
    # Information Tile
    # ----------------
    delta_t = time.perf_counter() - t_start
    textstr = '\n'.join((
        r'Information:',
        r'     Exercise: Energy Regression',
        r'     Group name: {}'.format(group_name),
        r'     Random Seed: {}'.format(seed),
        r'     # Cascade Events: {}'.format(num_cascade),
        r'     Background Level: {}'.format(background_level),
        r'     Runtime: {:3.3f}s'.format(delta_t),
        r'     Trained models: {}'.format(len(models)),
        r'',
        r'Tests',
        r'     Passed nan test: {}'.format(passed_nan_test),
        r'     Passed settings test: {}'.format(passed_settings_test),
        r'     Passed scaling test: {}'.format(passed_scaling_test),
    ))
    passed_all = (passed_nan_test & passed_settings_test & passed_scaling_test)
    if passed_all:
        edgecolor = 'green'
    else:
        edgecolor = 'red'
    props = dict(boxstyle='square', edgecolor=edgecolor, linewidth=2,
                 facecolor='white', alpha=0.5, pad=1)
    ax5.text(0.0, 0.95, textstr,
             transform=ax5.transAxes,
             linespacing=2.,
             verticalalignment='top',
             bbox=props)
    ax5.axis('Off')

    # ------------------
    # Save or show Plots
    # ------------------
    if output_directory is None:
        plt.show()
    else:
        plot_path = output_directory / 'energy_regression.pdf'
        fig.savefig(plot_path)


if __name__ == '__main__':
    main()
