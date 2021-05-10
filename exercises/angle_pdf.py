from matplotlib import pyplot as plt
import os
import numpy as np
import click
import time
import scipy.integrate as integrate

from utils import load_class, get_object


def run_test_for_various_k(test, rng, dist_class):
    """Run a test for various parameters.

    Parameters
    ----------
    test : callable
        The test to run.
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution
        The angle distribution class to use
    """
    for k in np.linspace(0.01, 10, 10):
        dist = dist_class(k)
        result = test(rng, dist)
        if not result:
            return False
    return True


def test_pdf(rng, dist):
    """Test some PDF values.

    Note: passing this test does not mean the PDF is correct!

    Parameters
    ----------
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution object
        The angle distribution object to use
    """
    num = 12
    delta_dir = np.random.uniform(low=-1, size=num)
    pdf_values = dist.pdf(delta_dir)

    # check length of returned values
    if len(pdf_values) != num:
        return False

    # values outside of [-pi, +pi) should be zero
    eps = 1e-15
    if not ((dist.pdf(-3*np.pi) == 0) and
            (dist.pdf(-np.pi - eps) == 0) and
            (dist.pdf(np.pi + eps) == 0) and
            (dist.pdf(2*np.pi) == 0)
            ):
        return False

    # values at 0. should be the larges
    if not ((dist.pdf(0.) > dist.pdf(0.1)) and
            (dist.pdf(0.) > dist.pdf(-0.1))
            ):
        return False

    return True


def test_pdf_integral(dist, upper_limit=np.inf, expected_value=1.0):
    """Check if integral over PDF produces expected values

    Parameters
    ----------
    dist : AngleDistribution object
        The angle distribution object to use
    upper_limit : float
        The upper limit of the integral
    expected_value : float
        The expected value

    Returns
    -------
    bool
        True if calculated matches expected value within error range
    """
    area, estimated_error = integrate.quad(dist.pdf, -np.inf, upper_limit)
    estimated_error = max(1e-3, estimated_error)
    return np.abs(expected_value - area) <= estimated_error


def test_normalization(rng, dist):
    """Test correct normalization of PDF

    Parameters
    ----------
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution object
        The angle distribution object to use
    """
    return test_pdf_integral(dist)


def test_cdf(rng, dist):
    """Test some CDF values.

    Note: passing this test does not mean the CDF is correct!

    Parameters
    ----------
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution object
        The angle distribution object to use
    """
    num = 12
    delta_dir = rng.uniform(low=-1, size=num)
    cdf_values = dist.cdf(delta_dir)

    # check length of returned values
    if len(cdf_values) != num:
        return False

    # values below -pi should be zero, above +pi should be 1
    if not (np.allclose(dist.cdf(-3*np.pi), 0) and
            np.allclose(dist.cdf(-np.pi), 0) and
            (dist.cdf(np.pi) == 1) and
            (dist.cdf(2*np.pi) == 1) and
            np.allclose(dist.cdf(0.), 0.5)
            ):
        return False

    # check if PDF and CDF match up
    for cdf in np.linspace(-np.pi, np.pi, 10):
        if not test_pdf_integral(dist, cdf, expected_value=dist.cdf(cdf)):
            return False

    return True


def test_ppf(rng, dist):
    """Test PPF

    Note: passing this test does not mean the PPF is correct!

    Parameters
    ----------
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution object
        The angle distribution object to use
    """

    num = 12
    q = np.random.uniform(size=num)
    cdf_values = dist.ppf(q)

    # check length of returned values
    if len(cdf_values) != num:
        return False

    # Check some values
    if not (np.allclose(dist.ppf(0.5), 0., rtol=1e-3, atol=1e-3) and
            np.allclose(dist.ppf(0.), -np.pi, rtol=1e-3, atol=1e-3) and
            np.allclose(dist.ppf(1.0), np.pi, rtol=1e-3, atol=1e-3)
            ):
        return False

    # check if PPF and CDF match up
    for q in np.linspace(0., 1., 10):
        if not test_pdf_integral(dist, dist.ppf(q), expected_value=q):
            return False

    return True


def test_rvs(rng, dist):
    """Test RVS

    Note: passing this test does not mean the rvs method is correct!

    Parameters
    ----------
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution object
        The angle distribution object to use
    """

    num = 1000
    delta_dir = dist.rvs(rng, num)

    # check length of returned values
    if len(delta_dir) != num:
        return False

    # values below -pi should be zero, above +pi should be 1
    eps = 1e-15
    if not (np.all(delta_dir >= -np.pi - eps) and
            np.all(delta_dir <= np.pi + eps)
            ):
        return False

    # check if something is obviously wrong
    fraction_below = np.sum(delta_dir < 0) / num
    if fraction_below < 0.3 or fraction_below > 0.7:
        return False

    return True


def test_reproducibily(rng, dist):
    """Test Reproducibility of RVS Method

    Parameters
    ----------
    rng : RNG object
        The random number generator object to use.
    dist : AngleDistribution object
        The angle distribution object to use
    """

    seed = rng.integers(0, 100)
    num = 1000
    rng1 = np.random.default_rng(seed)
    delta_dir1 = dist.rvs(rng1, num)

    rng2 = np.random.default_rng(seed)
    delta_dir2 = dist.rvs(rng2, num)

    return np.allclose(delta_dir1, delta_dir2)


@click.command()
@click.argument('group_name', type=str)
@click.option('--seed', '-s', default=1337, type=int,
              help='Random number generator seed')
@click.option('--n_samples', '-n', default=100000, type=int,
              help='Number of samples to draw for plots and tests.')
@click.option('--angle_pdf_k', '-k', default=None, type=float,
              help='If provided, another ExponentialDeltaAngleDistribution '
                   'will be added to the plots with this value set as k.')
@click.option('--compare/--no-compare', default=False,
              help='Compare to DeltaAngleDistribution which will be added to '
                   'plots if flag is set.')
@click.option('--output_directory', '-d', default=None,
              help='Folder were plots will be saved to')
def main(group_name, seed, n_samples, angle_pdf_k, compare, output_directory):

    # start timer
    t_start = time.perf_counter()

    # create random number generator
    rng = np.random.default_rng(seed)

    # load class
    dist_class = load_class(
        group_name + '.simulation.detector.ExponentialDeltaAngleDistribution'
    )

    # create angular distributions
    dist = get_object(
        group_name + '.simulation.detector.ExponentialDeltaAngleDistribution'
    )
    dist_01 = get_object(
        group_name + '.simulation.detector.ExponentialDeltaAngleDistribution',
        k=0.1,
    )
    dist_2 = get_object(
        group_name + '.simulation.detector.ExponentialDeltaAngleDistribution',
        k=2,
    )
    if angle_pdf_k:
        dist_k = get_object(
            (group_name +
             '.simulation.detector.ExponentialDeltaAngleDistribution'),
            k=angle_pdf_k,
        )
    if compare:
        dist_compare = get_object(
            group_name + '.simulation.detector.DeltaAngleDistribution'
        )

    # create plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))

    # ---------
    # Run Tests
    # ---------

    # check PDF
    passed_pdf_test = run_test_for_various_k(test_pdf, rng, dist_class)

    # check normalization of PDF
    passed_normalization_test = run_test_for_various_k(
        test_normalization, rng, dist_class
    )

    # check CDF
    passed_cdf_test = run_test_for_various_k(test_cdf, rng, dist_class)

    # check PPF
    passed_ppf_test = run_test_for_various_k(test_ppf, rng, dist_class)

    # check PPF
    passed_rvs_test = run_test_for_various_k(test_rvs, rng, dist_class)

    # test reproducibility of rvs
    passed_reproducibilty = run_test_for_various_k(
        test_reproducibily, rng, dist_class
    )

    passed_all = (passed_pdf_test & passed_normalization_test &
                  passed_cdf_test & passed_ppf_test & passed_rvs_test &
                  passed_reproducibilty)

    # --------
    # Draw PDF
    # --------
    x = np.linspace(-1.5*np.pi, 1.5*np.pi, 1000)
    axes[0, 0].plot(x, dist_01.pdf(x), label='k = 0.1')
    axes[0, 0].plot(x, dist.pdf(x), label='k = 1')
    axes[0, 0].plot(x, dist_2.pdf(x), label='k = 2')
    if angle_pdf_k:
        axes[0, 0].plot(x, dist_k.pdf(x), label='k = {}'.format(angle_pdf_k))
    if compare:
        axes[0, 0].plot(x, dist_compare.pdf(x), label='DeltaAngleDistribution')
    axes[0, 0].set_xlabel('Delta Angle / radians')
    axes[0, 0].set_ylabel('PDF')
    axes[0, 0].set_title('PDF')
    axes[0, 0].legend()

    # --------
    # Draw CDF
    # --------
    x = np.linspace(-1.5*np.pi, 1.5*np.pi, 1000)
    axes[0, 1].plot(x, dist_01.cdf(x), label='k = 0.1')
    axes[0, 1].plot(x, dist.cdf(x), label='k = 1')
    axes[0, 1].plot(x, dist_2.cdf(x), label='k = 2')
    if angle_pdf_k:
        axes[0, 1].plot(x, dist_k.cdf(x), label='k = {}'.format(angle_pdf_k))
    if compare:
        axes[0, 1].plot(x, dist_compare.cdf(x), label='DeltaAngleDistribution')
    axes[0, 1].set_xlabel('Delta Angle / radians')
    axes[0, 1].set_ylabel('CDF')
    axes[0, 1].set_title('CDF')
    axes[0, 1].legend()

    # --------
    # Draw PPF
    # --------
    x = np.linspace(-0., 1., 1000)
    axes[1, 0].plot(x, dist_01.ppf(x), label='k = 0.1')
    axes[1, 0].plot(x, dist.ppf(x), label='k = 1')
    axes[1, 0].plot(x, dist_2.ppf(x), label='k = 2')
    if angle_pdf_k:
        axes[1, 0].plot(x, dist_k.ppf(x), label='k = {}'.format(angle_pdf_k))
    if compare:
        axes[1, 0].plot(x, dist_compare.ppf(x), label='DeltaAngleDistribution')
    axes[1, 0].set_xlabel('Quantile')
    axes[1, 0].set_ylabel('PPF')
    axes[1, 0].set_title('PPF')
    axes[1, 0].legend()

    # --------
    # Draw RVS
    # --------
    bins = np.linspace(-1.1*np.pi, 1.1*np.pi, n_samples // 100)
    bin_mids = bins[:-1] + np.diff(bins)
    samples = dist.rvs(random_state=rng, size=n_samples)
    axes[1, 1].hist(samples, bins=bins, label='Sampled (k = 1)',
                    histtype='step', density=True)
    axes[1, 1].plot(bin_mids, dist.pdf(bin_mids), label='PDF')
    axes[1, 1].set_xlabel('Delta Angle / radians')
    axes[1, 1].set_ylabel('Relative Frequency')
    axes[1, 1].set_title('Samples')
    axes[1, 1].legend()

    # ----------------
    # Information Tile
    # ----------------
    delta_t = time.perf_counter() - t_start

    textstr = '\n'.join((
        r'Information:',
        r'     Exercise: Angle PDF',
        r'     Random Seed: {}'.format(seed),
        r'     Runtime: {:3.3f}s'.format(delta_t),
        r'     Group name: {}'.format(group_name),
        r'',
        r'Tests:',
        r'     Passed PDF test: {}'.format(passed_pdf_test),
        r'     Passed CDF test: {}'.format(passed_cdf_test),
        r'     Passed PPF test: {}'.format(passed_ppf_test),
        r'     Passed RVS test: {}'.format(passed_rvs_test),
        r'     PDF is normalized: {}'.format(passed_normalization_test),
        r'     Samples are reproducible: {}'.format(passed_reproducibilty),
        r'',
    ))
    axes[0, 2].axis('Off')
    axes[1, 2].axis('Off')

    # these are matplotlib.patch.Patch properties
    if passed_all:
        edgecolor = 'green'
    else:
        edgecolor = 'red'
    props = dict(boxstyle='square', edgecolor=edgecolor, linewidth=2,
                 facecolor='white', alpha=0.5, pad=1)

    axes[0, 2].text(0.0, 0.95, textstr,
                    transform=axes[0, 2].transAxes,
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
            'exercise_angle_pdf.pdf')
        plot_dir = os.path.dirname(plot_path)

        if not os.path.exists(plot_dir):
            print('Creating plot directory: {}'.format(plot_dir))
            os.makedirs(plot_dir)
        fig.savefig(plot_path)


if __name__ == '__main__':
    main()
