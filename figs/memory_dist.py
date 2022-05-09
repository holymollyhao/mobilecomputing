import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import ImageColor
import numpy as np
import tqdm

Distributions = {
    'Distribution 1': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    'Distribution 2': [191, 674, 51, 654, 1900, 20, 10, 200, 193, 246],
    'Distribution 3': [170, 72, 23, 12, 19, 76, 49, 230, 33, 203],
    'Distribution 4': [191, 253, 180, 252, 253, 200, 299, 150, 201, 345]
}

if __name__=='__main__':

    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams.update({'font.size': 20})

    with plt.style.context("seaborn-deep"):
        for i, data_key in enumerate(list(Distributions.keys())):
            fig = plt.figure(figsize=(5, 5))

            plt.grid(False)

            # plot data
            data = Distributions[data_key]
            bar_plot = plt.bar(range(len(data)), data, color='royalblue', width=0.8,
                    align='center', edgecolor='black', linewidth=1)
            # add values
            for idx, rect in enumerate(bar_plot):
                x = rect.get_x()
                h = rect.get_height()
                w = rect.get_width()
                plt.text(x + w/2, h+.5, data[idx], ha='center', va='bottom', fontweight='bold')

            # style
            plt.xticks(np.arange(10), np.arange(10))
            plt.title(data_key, pad=10)

            plt.xlabel('Class ID', labelpad=10)
            plt.ylim(0, max(data)*1.1)

            # remove right and left border
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            plt.gca().set_axisbelow(True)
            plt.tight_layout(pad=0.1)

            plt.savefig(f'memory_dist{i+1}.pdf')
            plt.show()