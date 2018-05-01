from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def vis_pc(pc, show=True, save_dir=None):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='.')
    ax.grid(False)
    # ax.axis('off')
    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)


if __name__ == '__main__':
    plt.figure()
    plt.plot(np.arange(1, 10))
    plt.show()