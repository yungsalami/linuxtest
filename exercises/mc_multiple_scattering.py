from matplotlib import pyplot as plt
import os
import numpy as np
import click
import time
from tqdm import tqdm

from project_a5.simulation.detector.angle_dist import DeltaAngleDistribution
from project_a5.simulation.detector.exp_angle_dist import (
    ExponentialDeltaAngleDistribution
)

from utils import get_object


def get_particle(group_name,
                 scattering_length=10.,
                 absorption_length=500.,
                 energy=1.,
                 direction=0.,
                 x=0.,
                 y=0.,
                 angle_distribution=DeltaAngleDistribution(),
                 **kwargs):
    """Get a multiple scattering particle.

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    scattering_length : float
        The scattering length to use.
    absorption_length : float
        The absorption length to use.
    energy : float, optional
        The energy of the energy depositions.
    direction : float, optional
        The direction of the particle
    x : float, optional
        The vertex x-coordinate of the particle.
    y : float, optional
        The vertex y-coordinate of the particle.
    angle_distribution : DeltaAnglePDF, optional
        The delta angle distribution to use to sample new delta direction
        vectors. Default distribution is: `DeltaAngleDistribution`.
    **kwargs
        Keyword arguments that will be passed on the particle constructor.

    Returns
    -------
    MultipleScatteringParticle object
        The created multiple scattering particle
    """
    particle_class = (group_name +
                      '.simulation.particle.MultipleScatteringParticle')
    particle = get_object(
        full_class_string=particle_class,
        energy=energy,
        direction=direction,
        x=x,
        y=y,
        scattering_length=scattering_length,
        absorption_length=absorption_length,
        angle_distribution=angle_distribution,
        **kwargs
    )
    return particle


def get_distances(depositions):
    """Get interaction distances from a given list of depositions

    Parameters
    ----------
    depositions : list of tuples
        A list of energy depositions as returned by the particle
        propagate method. A deposition consists of a tuple of:
            [x-position, y-position, deposited Energy, direction]

    Returns
    -------
    array_like
        The distances between interaction points.
    """
    dx = np.diff(depositions[:, 0])
    dy = np.diff(depositions[:, 1])
    distances = np.sqrt(dx**2 + dy**2)
    return distances


def get_directions(depositions):
    """Get directions from a given list of depositions

    Parameters
    ----------
    depositions : list of tuples
        A list of energy depositions as returned by the particle
        propagate method. A deposition consists of a tuple of:
            [x-position, y-position, deposited Energy, direction]

    Returns
    -------
    array_like
        The directions between interaction points.
    """
    dx = np.diff(depositions[:, 0])
    dy = np.diff(depositions[:, 1])
    directions = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
    return directions


def test_directions(group_name, rng, angle_distribution):
    """Test if directions of energy depositions are correct.

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    rng : RNG object
        The random number generator to use.
    angle_distribution : DeltaAnglePDF, optional
        The delta angle distribution to use to sample new delta direction
        vectors. Default distribution is: `DeltaAngleDistribution`.

    Returns
    -------
    bool
        True if test is passed.
    """
    for i in range(10):

        # draw scattering and absorption
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        energy = rng.uniform(1, 100)
        direction = rng.uniform(0, 2*np.pi)

        # get energy depositions
        depositions = np.asarray(get_particle(
            group_name=group_name,
            scattering_length=rng.uniform(10, 30),
            absorption_length=rng.uniform(100, 300),
            x=x,
            y=y,
            energy=energy,
            direction=direction,
            angle_distribution=angle_distribution,
        ).propagate(rng))

        # check correct shape
        if not (len(depositions) >= 2 and depositions.shape[1] == 4):
            return False

        # check correct values
        directions = get_directions(depositions)
        n_bounds = depositions[1:, 3].astype(int) + 1
        directions_dep = (depositions[1:, 3] + n_bounds*2*np.pi) % (2*np.pi)
        if not np.allclose(directions_dep, directions):
            return False

    return True


def test_reproducibility(group_name, rng, angle_distribution):
    """Test if particle propagation results are reproducible.

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    rng : RNG object
        The random number generator to use.
    angle_distribution : DeltaAnglePDF, optional
        The delta angle distribution to use to sample new delta direction
        vectors. Default distribution is: `DeltaAngleDistribution`.

    Returns
    -------
    bool
        True if test is passed.
    """
    seed = rng.integers(0, 100)

    for i in range(3):

        # draw scattering and absorption
        scat_length = rng.uniform(10, 30)
        abs_length = rng.uniform(100, 300)

        # get energy depositions
        depositions1 = np.asarray(get_particle(
            group_name=group_name,
            scattering_length=scat_length,
            absorption_length=abs_length,
            angle_distribution=angle_distribution,
        ).propagate(np.random.default_rng(seed + i)))

        depositions2 = np.asarray(get_particle(
            group_name=group_name,
            scattering_length=scat_length,
            absorption_length=abs_length,
            angle_distribution=angle_distribution,
        ).propagate(np.random.default_rng(seed + i)))

        # check correct shape
        if depositions1.shape != depositions2.shape:
            return False

        # check correct values
        if not np.allclose(depositions1, depositions2):
            return False

    return True


def test_vertex(group_name, rng, angle_distribution):
    """Test if first energy deposition is the particle vertex.

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    rng : RNG object
        The random number generator to use.
    angle_distribution : DeltaAnglePDF, optional
        The delta angle distribution to use to sample new delta direction
        vectors. Default distribution is: `DeltaAngleDistribution`.

    Returns
    -------
    bool
        True if test is passed.
    """
    for i in range(10):

        # draw scattering and absorption
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        energy = rng.uniform(1, 100)
        direction = rng.uniform(0, 2*np.pi)

        # get energy depositions
        depositions = np.asarray(get_particle(
            group_name=group_name,
            scattering_length=rng.uniform(10, 30),
            absorption_length=rng.uniform(100, 300),
            x=x,
            y=y,
            energy=energy,
            direction=direction,
            angle_distribution=angle_distribution,
        ).propagate(rng))

        # check correct shape
        if not (len(depositions) >= 2 and depositions.shape[1] == 4):
            return False

        # check correct values
        if not np.allclose(depositions[0], (x, y, 0., direction)):
            return False

    return True


def test_deposited_energies(group_name, rng, angle_distribution):
    """Test if the deposited energies are correct.

    All energy depositions at scattering points should be zero, because the
    particle scatters fully elastic. The last interaction point should be the
    point of absorption at which the particle losses all of its energy.

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    rng : RNG object
        The random number generator to use.
    angle_distribution : DeltaAnglePDF, optional
        The delta angle distribution to use to sample new delta direction
        vectors. Default distribution is: `DeltaAngleDistribution`.

    Returns
    -------
    bool
        True if test is passed.
    """
    for i in range(10):

        # draw scattering and absorption
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        energy = rng.uniform(1, 100)
        direction = rng.uniform(0, 2*np.pi)

        # get energy depositions
        depositions = np.asarray(get_particle(
            group_name=group_name,
            scattering_length=rng.uniform(10, 30),
            absorption_length=rng.uniform(100, 300),
            x=x,
            y=y,
            energy=energy,
            direction=direction,
            angle_distribution=angle_distribution,
        ).propagate(rng))

        # check correct shape
        if not (len(depositions) >= 2 and depositions.shape[1] == 4):
            return False

        # check correct values
        if not (np.allclose(0., depositions[:-1, 2]) and
                np.allclose(energy, depositions[-1, 2])):
            return False

    return True


def test_scattering_pdf(group_name, rng):
    """Test if the provided scattering pdf is used correctly.

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    rng : RNG object
        The random number generator to use.

    Returns
    -------
    bool
        True if test is passed.
    """

    class DummyAnglePDf():
        def rvs(self, random_state, size=None):
            if size is None:
                return 0.
            else:
                return np.zeros(size)

    dummy_angle_pdf = DummyAnglePDf()

    for i in range(10):

        # draw scattering and absorption
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        energy = rng.uniform(1, 100)
        direction = rng.uniform(0, 2*np.pi)

        # get energy depositions
        depositions = np.asarray(get_particle(
            group_name=group_name,
            scattering_length=rng.uniform(10, 30),
            absorption_length=rng.uniform(100, 300),
            x=x,
            y=y,
            energy=energy,
            direction=direction,
            angle_distribution=dummy_angle_pdf,
        ).propagate(rng))

        # check correct shape
        if not (len(depositions) >= 2 and depositions.shape[1] == 4):
            return False

        # check correct values
        distances = get_distances(depositions)
        dx = np.diff(depositions[:, 0])
        dy = np.diff(depositions[:, 1])
        dot_product = np.cos(direction) * dx + np.sin(direction) * dy
        if not (np.allclose(distances, dot_product) and
                np.allclose(0., depositions[:-1, 2]) and
                np.allclose(energy, depositions[-1, 2]) and
                np.allclose(direction, depositions[:, 3])):
            return False

    return True


def get_normalized_lengths(group_name, rng, angle_distribution, n=1000,):
    """Get normalized absorption and scattering lengths

    Parameters
    ----------
    group_name : str
        The name of the module/group to use.
    rng : RNG object
        The random number generator to use.
    angle_distribution : DeltaAnglePDF, optional
        The delta angle distribution to use to sample new delta direction
        vectors. Default distribution is: `DeltaAngleDistribution`.
    n : int, optional
        Description

    Returns
    -------
    array_like
        The normalized absorption lengths
    array_like
        The normalized scattering lengths
    """

    # sample random absorption lengths
    absorption_lengths = rng.uniform(100, 150, size=n)
    scattering_lengths = rng.uniform(10, 20,  size=n)

    norm_abs_lengths = []
    norm_scat_lengths = []

    for abs_length, scat_length in tqdm(zip(absorption_lengths,
                                            scattering_lengths),
                                        total=n):

        # get particle
        particle = get_particle(
            group_name=group_name,
            scattering_length=scat_length,
            absorption_length=abs_length,
            angle_distribution=angle_distribution,
        )

        # propagate and obtain energy depositions
        depositions = np.asarray(particle.propagate(rng))

        # compute distances between interactions
        distances = get_distances(depositions)

        # last distance is absorption, so it does not count for scattering
        scat_distances = np.array(distances[:-1])
        abs_distance = np.sum(distances)

        # compute and append normalized distances
        norm_abs_lengths.append(abs_distance/abs_length)
        if len(scat_distances) > 0:
            norm_scat_lengths.extend(scat_distances/scat_length)

    return norm_abs_lengths, norm_scat_lengths


def draw_normalized_lengths(ax, normalized_lengths, rng):
    """Draw the normalized lengths.

    Parameters
    ----------
    ax : axis
        The axis to use.
    normalized_lengths : array_like
        The normalized lengths to draw.
    rng : RNG object
        The random number generator to use.
    """
    bins = np.linspace(0, 10, 100)
    ax.hist(
        normalized_lengths,
        bins=bins,
        histtype='step',
        density=True,
        label='Simulation',
    )
    ax.hist(
        rng.exponential(size=100000),
        bins=bins,
        histtype='step',
        density=True,
        label='Expectation',
    )
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Normalized Lenghts')
    ax.set_ylabel('Relative Frequency')


def draw_depositions(ax, depositions):
    """Draw energy depositions on provided axis

    Parameters
    ----------
    ax : axis
        The axis to use.
    depositions : list of tuple
        The energy depositions to draw.
        A deposition consists of a tuple of:
            [x-position, y-position, deposited Energy, direction]
    """
    distances = get_distances(depositions)
    propagation_distance = np.sum(distances)
    num_interactions = len(distances)

    ax.scatter(depositions[:, 0], depositions[:, 1],
               label='Interactions: {}'.format(num_interactions))
    ax.plot(depositions[:, 0], depositions[:, 1],
            label='Distance: {:2.2f}'.format(propagation_distance))
    ax.set_xlabel('X-Coordinate / arb. units')
    ax.set_ylabel('Y-Coordinate / arb. units')
    ax.legend()


@click.command()
@click.argument('group_name', type=str)
@click.option('--seed', '-s', default=1337, type=int,
              help='Random number generator seed')
@click.option('--num_repetitions', '-n', default=1000, type=int,
              help='Number of repetitions for exponential test.')
@click.option('--angle_pdf_k', '-k', default=None, type=float,
              help='The parameter k of the ExponentialDeltaAngleDistribution '
                   'to use. If None is provided, DeltaAngleDistribution will '
                   'be used as the delta angle pdf')
@click.option('--output_directory', '-d', default=None,
              help='Folder were plots will be saved to')
def main(group_name, seed, num_repetitions, angle_pdf_k, output_directory):

    # start timer
    t_start = time.perf_counter()

    # choose the DeltaAngleDistribution
    if angle_pdf_k is None:
        angle_distribution = DeltaAngleDistribution()
        angle_str = 'DeltaAngleDistribution'
    else:
        angle_distribution = ExponentialDeltaAngleDistribution(angle_pdf_k)
        angle_str = 'ExpDeltaAngleDistribution (k={:2.2f})'.format(
            angle_pdf_k)

    # create random number generator
    rng = np.random.default_rng(seed)

    # create plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))

    # ----------------------
    # Normal scattering plot
    # ----------------------
    particle = get_particle(group_name, angle_distribution=angle_distribution)
    depositions = np.asarray(particle.propagate(rng))
    draw_depositions(axes[0, 0], depositions)
    axes[0, 0].set_title('Normal scattering')

    # --------------------
    # Less scattering plot
    # --------------------
    particle = get_particle(
        group_name,
        scattering_length=100.,
        angle_distribution=angle_distribution,
    )
    depositions = np.asarray(particle.propagate(rng))
    draw_depositions(axes[0, 1], depositions)
    axes[0, 1].set_title('Less scattering expected')

    # ------------------
    # No scattering plot
    # ------------------
    particle = get_particle(
        group_name,
        scattering_length=1e16,
        angle_distribution=angle_distribution,
    )
    depositions = np.asarray(particle.propagate(rng))
    draw_depositions(axes[0, 2], depositions)
    axes[0, 2].set_title('No scattering expected')

    # ---------
    # Run Tests
    # ---------
    norm_abs_lengths, norm_scat_lengths = get_normalized_lengths(
        group_name, rng, angle_distribution, n=num_repetitions,
    )

    # check if directions of depositions are correct
    passed_direction_test = test_directions(
        group_name, rng, angle_distribution=angle_distribution,
    )

    # check if results are reproducible
    passed_reproducibilty = test_reproducibility(
        group_name, rng, angle_distribution=angle_distribution,
    )

    # check if the first interaction poit is the vertex of the particle
    passed_vertex_test = test_vertex(
        group_name, rng, angle_distribution=angle_distribution,
    )

    # check if the particle is using the provided angle distribution pdf
    passed_scattering_pdf = test_scattering_pdf(group_name, rng)

    # check if amount of deposited energy is correct
    passed_energy_test = test_deposited_energies(
        group_name, rng, angle_distribution=angle_distribution,
    )

    passed_all = (passed_direction_test & passed_vertex_test &
                  passed_scattering_pdf & passed_reproducibilty &
                  passed_energy_test)

    # -----------------------
    # Absorption lengths plot
    # -----------------------
    draw_normalized_lengths(axes[1, 0], norm_abs_lengths, rng)
    axes[1, 0].set_xlabel('Normalized Absorption Lenghts')

    # -----------------------
    # Scattering lengths plot
    # -----------------------
    draw_normalized_lengths(axes[1, 1], norm_scat_lengths, rng)
    axes[1, 1].set_xlabel('Normalized Scattering Lenghts')

    # ----------------
    # Information Tile
    # ----------------
    delta_t = time.perf_counter() - t_start

    textstr = '\n'.join((
        r'Information:',
        r'     Exercise: MC Multiple Scattering Particle',
        r'     Angle PDF: {}'.format(angle_str),
        r'     Num Repetitions: {}'.format(num_repetitions),
        r'     Random Seed: {}'.format(seed),
        r'     Runtime: {:3.3f}s'.format(delta_t),
        r'     Group name: {}'.format(group_name),
        r'',
        r'Tests:',
        r'     Passed directions test: {}'.format(passed_direction_test),
        r'     Results are reproducible: {}'.format(passed_reproducibilty),
        r'     First interaction is vertex: {}'.format(passed_vertex_test),
        r'     Passed scattering pdf test: {}'.format(passed_scattering_pdf),
        r'     Deposited energy is correct: {}'.format(passed_energy_test),
        r'',
    ))
    axes[1, 2].axis('Off')

    # these are matplotlib.patch.Patch properties
    if passed_all:
        edgecolor = 'green'
    else:
        edgecolor = 'red'
    props = dict(boxstyle='square', edgecolor=edgecolor, linewidth=2,
                 facecolor='white', alpha=0.5, pad=1)

    axes[1, 2].text(-0.2, 0.95, textstr,
                    transform=axes[1, 2].transAxes,
                    linespacing=2.,
                    verticalalignment='top',
                    bbox=props)
    # ----------------

    fig.tight_layout()
    if output_directory is None:
        plt.show()
    else:
        plot_path = os.path.join(
            output_directory,
            group_name,
            'exercise_mc_multiple_scattering.pdf')
        plot_dir = os.path.dirname(plot_path)

        if not os.path.exists(plot_dir):
            print('Creating plot directory: {}'.format(plot_dir))
            os.makedirs(plot_dir)
        fig.savefig(plot_path)


if __name__ == '__main__':
    main()
