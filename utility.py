import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def side_by_side_plot_generator(img1, img2, figure_length, figure_width, title, orientation, title1=None, title2=None, loc1=None,
                                loc2=None, dpi=None):
    if orientation == 'horizontal':
        fig, ax = plt.subplots(1, 2, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0, 0.5, 1])
        ax[1].set_position([0.5, 0, 0.5, 1])
    if orientation == 'vertical':
        fig, ax = plt.subplots(2, 1, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0.5, 1, 0.5])
        ax[1].set_position([0, 0, 1, 0.5])
        # ax[0].set_position([0, 0.33, 1, 0.66])
        # ax[1].set_position([0, 0, 1, 0.33])
    # remove the borders
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(0)
        ax[1].spines[axis].set_linewidth(0)
    ax[0].imshow(img1)
    ax[0].set_title(title1, loc=loc1)
    # remove the x and y ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(img2)
    ax[1].set_title(title2, loc=loc2)
    # remove the x and y ticks
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # for i in range(2):
    #     ax[i].title.set_fontsize(8)
    # save the figure
    plt_name = title + '.png'
    plt_dir = './Figures/'
    plt_path = plt_dir + plt_name
    plt.savefig(plt_path)
    plt.show()