from matplotlib import pyplot as plt
import click
import time
import numpy as np
from scipy.optimize import curve_fit

from utils import load_class


template = '''Information:
  Exercise: Verteilungen
  Group name: {group_name}
Runtime:
  exp:    {time_exponential:.1f} ms (ref: 3.2 ms)
  power:  {time_power:.1f} ms (ref: 2.6 ms)
  cauchy: {time_cauchy:.1f} ms (ref: 3.1 ms)
Tests:
  exp mean correct: {exp_correct}
  power n correct:  {power_correct}
'''


def power(x, x_min, x_max, n):
    exp = 1 - n
    norm = -exp / (x_min**exp - x_max**exp)
    return norm * x**(-n)


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

    fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(10, 10))

    gen = Generator(seed=0)
    n_samples = 100_000

    # exponential
    time_exponential = 0
    exp_correct = True
    for tau, ax in zip([5, 10], axs[0, :]):
        t0 = time.perf_counter()
        numbers = gen.exponential(tau, n_samples)
        time_exponential += time.perf_counter() - t0

        exp_correct &= np.isclose(np.mean(numbers), tau, rtol=0.01)

        ax.hist(numbers, bins=50, range=[0, 6 * tau], density=True)
        t = np.linspace(0, 6 * tau, 500)
        ax.plot(t, 1/tau * np.exp(-t/tau))
        ax.set_title(f'Exp.-Verteilung, τ={tau}')
        ax.set_yscale('log')
    tests.append(exp_correct)

    # power
    time_power = 0
    power_correct = True
    for x_min, x_max, n, ax in zip([2e2, 3e2], [1.5e3, 15e3], [2.0, 2.7], axs[1, :]):
        t0 = time.perf_counter()
        numbers = gen.power(n=n, x_min=x_min, x_max=x_max, size=n_samples)
        time_power += time.perf_counter() - t0

        bins = np.logspace(np.log10(x_min), np.log10(x_max), 51)
        hist, _, _ = ax.hist(numbers, bins=bins, density=True)

        centers = 0.5 * (bins[:-1] + bins[1:])

        try:
            params, cov = curve_fit(power, centers, hist, [100.0, 1000.0, 2.0])
            power_correct &= np.isclose(params[2], n, rtol=0.01)
        except Exception as e:
            print(f'Fit failed: {e}')
            power_correct = False

        ax.set_xscale('log')
        ax.set_yscale('log')

        x = np.logspace(np.log10(x_min), np.log10(x_max), 500)
        ax.plot(x, power(x, x_min, x_max, n))
        ax.set_title(fr'Potenz,$x_\mathrm{{min}} = {x_min}$')

    tests.append(power_correct)

    # cauchy

    t0 = time.perf_counter()
    numbers = gen.cauchy(n_samples)
    time_cauchy = time.perf_counter() - t0

    passed_all = all(tests)
    text = template.format(
        group_name=group_name,
        time_power=time_power * 1e3,
        time_exponential=time_exponential * 1e3,
        time_cauchy=time_cauchy * 1e3,
        exp_correct=exp_correct,
        power_correct=power_correct,
    )

    ax = axs[2, 0]
    ax.hist(numbers, bins=51, range=[-10, 10], density=True)
    x = np.linspace(-10, 10, 500)
    ax.plot(x, 1 / (np.pi * (1 + x**2)))
    ax.set_title('Cauchy')

    props = dict(
        boxstyle='square',
        edgecolor='green' if passed_all else 'red',
        linewidth=2,
        facecolor='white',
        alpha=0.5,
        pad=1
    )

    ax = axs[2, 1]
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
