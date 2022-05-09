import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import ImageColor
import numpy as np
import tqdm
import color_conf

Performance = {
    'i.i.d.': {
        'BNstats': 87.01,
        'PL': 65.65,
        'Tent': 79.83,
        'T3A': 67.68,
        'CoTTA': 69.57,
        'Ours': 56.53
    },
    'Non-i.i.d.': {
        'BNstats': 38.79,
        'PL': 40.32,
        'Tent': 37.23,
        'T3A': 36.55,
        'CoTTA': 32.49,
        'Ours': 57.95
    },
}

Src = 45.5

if __name__=='__main__':

    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams.update({'font.size': 20})

    methods = list(Performance['i.i.d.'].keys())

    with plt.style.context("seaborn-deep"):
        fig = plt.figure(figsize=(6, 5))
        plt.grid(True, alpha=0.8, linestyle=':')

        # plot data
        n_bars = len(methods)
        bar_width = 0.12
        x = np.arange(2)
        for i, method in enumerate(methods):
            data = []
            for setting in Performance:
                data.append(Performance[setting][method])
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
            plt.bar(x=x+x_offset, height=data, width=bar_width,
                    capsize=5,
                    ecolor='black',
                    label=method,
                    color=color_conf.color_conf[method], edgecolor="black", zorder=2)

        # src plot
        plt.axhline(y=Src, color='gray', linestyle='--', linewidth=2, label='Source', zorder=1)

        # style
        plt.title('Performance Overview')
        plt.xticks(x, list(Performance.keys()))
        plt.ylabel('Accuracy(%)')
        plt.ylim([0, 100])

        plt.legend(loc='best', edgecolor='black', ncol=2, fancybox=False)

        plt.gca().set_axisbelow(True)
        plt.tight_layout(pad=0.1)

        plt.savefig(f'performance_overview.pdf')

        plt.show()
