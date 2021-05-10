from matplotlib import pyplot as plt
import click
import time
import numpy as np

from utils import load_class


template = '''Information:
  Exercise: Linear-Kongruent
  Group name: {group_name}

Tests:
  m=16:            {lcg16_correct}
  std. uniform:    {std_uniform_test}
  uniform [-1, 2]: {std_uniform_test}
'''


@click.command()
@click.argument('group_name', type=str)
@click.option(
    '--output', '-o', default=None,
    help='Name for the plot output. If not given, plt.show() is called'
)
def main(group_name, output):
    tests = []

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # load class
    LCG = load_class(
        group_name + '.random.LCG'
    )

    lcg16_chain = np.array([7, 10, 9, 4, 11, 14, 13, 8, 15, 2, 1, 12, 3, 6, 5, 0])
    lcg16 = LCG(seed=0, a=5, c=7, m=16)
    lcg16_correct = np.allclose(lcg16.random_raw(16), lcg16_chain)
    tests.append(lcg16_correct)

    n_samples = 10000
    n_bins = 50
    lcg = LCG()

    # test standard uniforn
    numbers = lcg.uniform(size=n_samples)
    hist, _, _ = ax1.hist(numbers, range=[0, 1], bins=50)
    ax1.axhline(n_samples / n_bins, color='C1')
    ax1.set_xlim(0, 1)

    # make sure all bins have entries, no values are outside of range
    std_uniform_test = hist.sum() == n_samples and np.count_nonzero(hist) == n_bins
    tests.append(std_uniform_test)

    # test other range
    numbers = lcg.uniform(-1, 2, size=n_samples)
    hist, _, _ = ax2.hist(numbers, range=[-1, 2], bins=50)
    ax2.axhline(n_samples / n_bins, color='C1')
    ax2.set_xlim(-1, 2)

    # make sure all bins have entries, no values are outside of range
    uniform_test = hist.sum() == n_samples and np.count_nonzero(hist) == n_bins
    tests.append(uniform_test)

    passed_all = all(tests)
    text = template.format(
        group_name=group_name,
        lcg16_correct=lcg16_correct,
        std_uniform_test=std_uniform_test,
        uniform_test=uniform_test,
    )

    props = dict(
        boxstyle='square',
        edgecolor='green' if passed_all else 'red',
        linewidth=2,
        facecolor='white',
        alpha=0.5,
        pad=1
    )

    ax0.text(
        0.05, 0.95, text,
        transform=ax0.transAxes,
        linespacing=2.,
        verticalalignment='top',
        bbox=props,
        family='monospace'
    )
    ax0.set_axis_off()

    if output is None:
        plt.show()
    else:
        fig.savefig(output, dpi=300)


if __name__ == '__main__':
    main()
