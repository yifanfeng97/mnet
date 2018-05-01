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


def trans_std_file(src_filename, des_filename):

    num_verts, num_edge, num_faces, points, faces = utils.read_mesh(src_filename)
    utils.write_mesh(des_filename, num_verts, num_edge, num_faces, points, faces)

    # points = []
    # faces = []
    # with open(src_filename, 'r') as f:
    #     line = f.readline().strip()
    #     if line == 'OFF':
    #         num_verts, num_faces, num_edge = f.readline().split()
    #         num_verts = int(num_verts)
    #         num_faces = int(num_faces)
    #         num_edge = int(num_edge)
    #     else:
    #         num_verts, num_faces, num_edge = line[3:].split()
    #         num_verts = int(num_verts)
    #         num_faces = int(num_faces)
    #         num_edge = int(num_edge)
    #
    #     for idx in range(num_verts):
    #         line = f.readline()
    #         point = [float(v) for v in line.split()]
    #         points.append(point)
    #
    #     for idx in range(num_faces):
    #         line = f.readline()
    #         face = [int(t_f) for t_f in line.split()]
    #         faces.append(face)
    #
    #
    # file_content = 'OFF\n%d %d %d\n'%(num_verts, num_faces, num_edge)
    # for idx in range(len(points)):
    #     file_content += '%f %f %f\n'%(points[idx][0], points[idx][1], points[idx][2])
    # for idx in range(len(faces)):
    #     file_content += '%d %d %d %d\n'%(faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3])
    #
    # with open(des_filename, 'w') as f:
    #     f.write(file_content)


def generate(cfg):
    shape_all = glob.glob(osp.join(cfg.data_mesh_root, '*', '*', '*.off'))
    for shape in tqdm(shape_all):
        # if shape.find('bed_0039') == -1:
        #     continue
        class_name, set_name, file_name = get_info(shape)
        new_folder = osp.join(cfg.data_std_mesh_root, class_name, set_name)
        new_dir = osp.join(new_folder, file_name + '.off')
        if not osp.exists(new_folder):
            os.makedirs(new_folder)
        trans_std_file(shape, new_dir)


if __name__ == '__main__':
    # file_name = '/home/fengyifan/data/ModelNet40/dresser/test/dresser_0208.off'
    # des_name = '/home/fengyifan/data/dresser_0208.off'
    # trans_std_file(file_name, des_name)
    # print('done!')

    cfg = utils.config('../config/config.cfg')
    generate(cfg)
    print('done!')

