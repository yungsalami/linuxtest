from matplotlib import pyplot as plt
import click
import time
import numpy as np
from scipy.optimize import curve_fit

from utils import load_class


template = '''Information:
  Exercise: Polarmethode
  Group name: {group_name}
Runtime:
  {time_polar:.1f} ms (ref: 12.6 ms)
Tests:
  size correct: {polar_num_samps_correct}
  mean correct: {polar_mean_correct}
  std correct: {polar_std_correct}
'''


@click.command()
@click.argument('group_name', type=str)
@click.option(
    '--output', '-o', default=None,
    help='Name for the plot output. If not given, plt.show() is called'
)
def main(group_name, output):
    tests = []
    # load class
    Generator = load_class(
        group_name + '.random.Generator'
    )

    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 5))

    gen = Generator(seed=0)
    n_samples = 100000

    time_polar = 0
    polar_num_samps_correct = True
    polar_mean_correct = True
    polar_std_correct = True

    for mu, sigma, ax in zip([0, 5], [1, 3], axs[:2]):
        t0 = time.perf_counter()
        if mu == 0 and sigma == 1:
            numbers = gen.standard_normal(n_samples)
        else:
            numbers = gen.normal(mu, sigma, n_samples)
        time_polar += time.perf_counter() - t0

        polar_num_samps_correct &= np.isclose(len(numbers), n_samples, atol=0)
        polar_mean_correct &= np.isclose(np.mean(numbers), mu, atol=0.1)
        polar_std_correct &= np.isclose(np.std(numbers), sigma, rtol=0.01)

        ax.hist(numbers, bins=50, range=[-4 * sigma + mu, 4 * sigma + mu], density=True)
        x = np.linspace(-4 * sigma + mu, 4 * sigma + mu, 500)
        ax.plot(x, 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2)))
        ax.set_title(f'Normal Distribution, µ={mu}, σ={sigma}')
        tests.append(polar_num_samps_correct)
        tests.append(polar_mean_correct)
        tests.append(polar_std_correct)

    passed_all = all(tests)
    text = template.format(
        group_name=group_name,
        time_polar=time_polar * 1e3,
        polar_num_samps_correct=polar_num_samps_correct,
        polar_mean_correct=polar_mean_correct,
        polar_std_correct=polar_std_correct,
    )

    props = dict(
        boxstyle='square',
        edgecolor='green' if passed_all else 'red',
        linewidth=2,
        facecolor='white',
        alpha=0.5,
        pad=1
    )

    ax = axs[2]
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        linespacing=2.,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )
    ax.set_axis_off()

    if output is None:
        plt.show()
    else:
        fig.savefig(output, dpi=300)


if __name__ == '__main__':
    main()
