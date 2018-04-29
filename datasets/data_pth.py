import sys
sys.path.append('../')
import torch
import pickle
import utils
import os.path as osp
import random
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset



class MeshData(Dataset):
    def __init__(self, state='train') -> None:
        super().__init__()
        cfg = utils.config()
        self.cfg = cfg
        if not osp.exists(cfg.split_train) or not osp.exists(cfg.split_test):
            utils.split_meshes()
        if state is 'train':
            with open(cfg.split_train, 'rb') as f:
                self.shape_list = pickle.load(f)
        else:
            with open(cfg.split_test, 'rb') as f:
                self.shape_list = pickle.load(f)

        print('%s data num: %d'%(state, len(self.shape_list)))

    @staticmethod
    def normalize(data):
        rowsum = np.array(data.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sparse.diags(r_inv)
        data = r_mat_inv.dot(data)
        return data

    def __getitem__(self, index):
        shape = self.shape_list[index]
        num_verts, num_edge, num_faces, points, faces = utils.read_mesh(shape['mesh'])
        feature = np.array(points)
        label = shape['label']
        adj = np.eye(num_verts)
        for _i in range(num_faces):
            adj[faces[_i][1], faces[_i][2]], adj[faces[_i][2], faces[_i][1]] = 1, 1
            adj[faces[_i][1], faces[_i][3]], adj[faces[_i][3], faces[_i][1]] = 1, 1
            adj[faces[_i][2], faces[_i][3]], adj[faces[_i][3], faces[_i][2]] = 1, 1
        feature = self.normalize(feature)
        adj = self.normalize(adj)
        return torch.Tensor(feature), torch.Tensor(adj), torch.Tensor([label]).long()

    def __len__(self):
        return len(self.shape_list)


if __name__ == '__main__':
    m = MeshData('train')
    print(len(m))
    tmp = m[0]
    print(tmp[2])

