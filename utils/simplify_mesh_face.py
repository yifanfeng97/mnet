import os
import os.path as osp
import glob
from random import shuffle
from math import isnan
import numpy as np
from tqdm import tqdm
# from .config import config
import sys
sys.path.append('../')
import utils
# import matplotlib.pyplot as plt
# import bisect
# from mpl_toolkits.mplot3d import Axes3D


def get_info(shape_dir):
    splits = shape_dir.split('/')
    class_name = splits[-3]
    set_name = splits[-2]
    file_name = splits[-1].split('.')[0]
    return class_name, set_name, file_name


def generate(cfg, xml_file='mesh_simplify_1024_face.mlx'):
    shape_all = glob.glob(osp.join(cfg.data_std_mesh_root, '*', '*', '*.off'))
    for shape in tqdm(shape_all):
        # if shape.find('bed_0039') == -1:
        #     continue
        class_name, set_name, file_name = get_info(shape)
        new_folder = osp.join(cfg.data_simply_mesh_root, class_name, set_name)
        new_dir = osp.join(new_folder, file_name + '.off')
        if not osp.exists(new_folder):
            os.makedirs(new_folder)
        cmd = 'meshlabserver -i %s -o %s -s %s'%(shape, new_dir, xml_file)
        os.system(cmd)


def draw_pc(pc, show=True, save_dir=None):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='.')
    ax.grid(False)
    ax.axis('off')
    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)

if __name__ == '__main__':
    # file_name = '/home/fengyifan/data/ModelNet40/dresser/test/dresser_0208.off'
    # des_name = '/home/fengyifan/data/dresser_0208.off'
    # trans_std_file(file_name, des_name)
    # print('done!')

    cfg = utils.config('../config/config.cfg')
    generate(cfg)
    print('done!')

