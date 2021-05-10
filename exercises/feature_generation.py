import click
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

from project_a5.simulation import Detector, VertexParticleGenerator
from project_a5.simulation.particle import TrackParticle, CascadeParticle
from project_a5.reconstruction.preprocessing import FeatureGenerator


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
        num_track,
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
    num_track: int
        Number of simulated track events
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
            & (num_track == simulation_defaults['num_track'])
            & (background_level == simulation_defaults['background_level'])
            & (energy_low == simulation_defaults['energy_low'])
            & (energy_high == simulation_defaults['energy_high'])
            & (random_seed == simulation_defaults['random_seed'])
            & (gamma == simulation_defaults['gamma'])
            )


def simulate_particles(
        detector,
        random_generator,
        num_cascade,
        num_track,
        energy_low,
        energy_high,
        gamma
        ):

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
        name='DiffuseTracks',
        particle_args={
            'prior_propagation_distance': 0,
            'stochastic_relative_de_dx': (0.02, 0.2)
            },
    )

    events = []
    event_classes = []

    print('Simulating Cascade Events')
    for event_id in tqdm(range(num_cascade)):
        particle = diff_cascade_generator.generate(
                1,
                random_state=random_generator,
        )
        # simulate and append event
        events.append(detector.generate_event(
            particle,
            random_state=random_generator
        ))
        event_classes.append(0)

    print('Simulating Track Events')
    for event_id in tqdm(range(num_track)):
        particle = diff_track_generator.generate(
                1,
                random_state=random_generator
        )
        # simulate and append event
        events.append(detector.generate_event(
            particle,
            random_state=random_generator
        ))
        event_classes.append(1)

    return events, event_classes


def analyze_events(events, FG):
    analysed_events = []
    for event in events:
        analysed_events.append(FG.analyse(event))

    return pd.DataFrame(analysed_events)


@click.command()
@click.argument('group_name', type=str)
@click.option('--seed', '-s', default=1337, type=int,
              help='Random number generator seed')
@click.option('--num_cascade', '-nc',
              default=simulation_defaults['num_cascade'], type=int,
              help='Number of cascade events to simulate')
@click.option('--num_track', '-nt',
              default=simulation_defaults['num_track'], type=int,
              help='Number of track events to simulate')
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
def main(
        group_name,
        seed,
        num_cascade,
        num_track,
        energy_low,
        energy_high,
        gamma,
        background_level,
        output_directory):

    # start timer
    t_start = time.perf_counter()

    # set random generator
    rg = np.random.default_rng(seed)

    # ---------------------------
    # Simulate and analyse events
    # ---------------------------
    detector = Detector(
            noise_level=background_level,
            trigger_kernel_shape=[4, 4])
    FG = FeatureGenerator(detector)
    events, event_classes = simulate_particles(
            detector,
            rg,
            num_cascade,
            num_track,
            energy_low,
            energy_high,
            gamma,
            )

    print('Analysing events ...')
    X = analyze_events(events, FG)
    y = np.array(event_classes)

    # -------------
    # Test Features
    # -------------
    passed_nan_test = run_test_for_nan_values(X)

    # -------------
    # Test Settings
    # -------------
    passed_settings_test = run_test_for_proper_settings(
            num_cascade,
            num_track,
            energy_low,
            energy_high,
            gamma,
            background_level,
            seed)

    # --------------------------------------
    # Train the model using cross-validation
    # --------------------------------------
    model = RandomForestClassifier(
            n_estimators=20,
            max_features='sqrt',
            max_depth=10)
    model.random_state = seed

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, shuffle=True)

    print('Training model ...')
    # obtain trained models and scores
    model.fit(X_train, y_train)
    y_predict = model.predict_proba(X_test)
    y_predict_binary = (y_predict[:, 1] > 0.5).astype('int')
    # obtain trained models and scores

    # -------------
    # Create Figure
    # -------------
    print('Generating figures ...')
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax2)
    ax4 = fig.add_subplot(gs[1:, 1])
    ax5 = fig.add_subplot(gs[0, 1])

    # -----------------------------
    # Draw Prediction Probabilities
    # -----------------------------
    cascade_mask = np.where(y_test == 0)
    track_mask = np.where(y_test == 1)
    # results are tuples for each event -> (prob cascade, prob_track)
    ax1.hist(
            y_predict[cascade_mask][:, 0],
            label='Cascade Events',
            histtype='step',
            bins=20,
    )
    ax1.hist(
            y_predict[track_mask][:, 1],
            label='Track Events',
            histtype='step',
            bins=20
    )
    ax1.set_ylabel('Counts')
    ax1.set_xlabel('Prediction Score')
    ax1.set_title('Prediction Probabilities')
    ax1.legend()

    # -------------------------
    # Draw Precision and Recall
    # -------------------------
    thresholds = np.arange(21)/20
    precisions = []
    recalls = []
    for threshold in thresholds:
        predicted_class = (y_predict[:, 1] >= threshold).astype('int')
        precisions.append(metrics.precision_score(y_test, predicted_class))
        recalls.append(metrics.recall_score(y_test, predicted_class))
    ax2.plot(thresholds, precisions, label='Precision')
    ax2.plot(thresholds, recalls, label='Recall')
    ax2.set_xlabel('Prediction Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision and Recall')
    ax2.legend()

    # ---------------------
    # Generate more metrics
    # ---------------------
    mean_fpr, mean_tpr, _ = metrics.roc_curve(y_test, y_predict[:, 1])
    roc_auc = metrics.roc_auc_score(y_test, y_predict[:, 1])
    ax3.plot(mean_fpr, mean_tpr, label='ROC (AUC = {:2.3f})'.format(roc_auc))
    ax3.set_title('Mean ROC-Curve')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc="lower right")

    # ------------------------
    # Draw Feature Importances
    # ------------------------
    feature_importances = np.array([
            tree.feature_importances_
            for tree in model])
    order = np.argsort(np.median(feature_importances, axis=0))
    ax4.boxplot(
            feature_importances[:, order],
            vert=False,
            sym='',
            medianprops={'color': 'C0'}
    )
    ax4.set_xlabel('Feature Importance')
    ax4.set_title('Feature Importances')
    ax4.set_yticklabels(X.columns.values[order])

    # ----------------
    # Information Tile
    # ----------------
    delta_t = time.perf_counter() - t_start
    textstr = '\n'.join((
        r'Information:',
        r'     Exercise: Feature Generation',
        r'     Group name: {}'.format(group_name),
        r'     Random Seed: {}'.format(seed),
        r'     # Track Events: {}'.format(num_track),
        r'     # Cascade Events: {}'.format(num_cascade),
        r'     Background Level: {}'.format(background_level),
        r'     Runtime: {:3.3f}s'.format(delta_t),
        r'     Test Accuracy: {0:.2f}'.format(
            metrics.accuracy_score(y_test, y_predict_binary)),
        r'',
        r'Tests',
        r'     Passed nan test: {}'.format(passed_nan_test),
        r'     Passed settings test: {}'.format(passed_settings_test),
    ))
    passed_all = (passed_nan_test & passed_settings_test)
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
        plot_path = os.path.join(
            output_directory,
            group_name,
            'feature_generation.pdf')
        plot_dir = os.path.dirname(plot_path)

        if not os.path.exists(plot_dir):
            print('Creating plot directory: {}'.format(plot_dir))
            os.makedirs(plot_dir)
        fig.savefig(plot_path)


if __name__ == '__main__':
    main()
