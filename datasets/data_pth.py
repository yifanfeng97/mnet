import sys
sys.path.append('../')
import torch
import pickle
import utils
import os.path as osp
import random
import os
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset

# bad case curtain 0066


class MeshData(Dataset):
    def __init__(self, cfg, state='train') -> None:
        super().__init__()
        self.cfg = cfg
        if not osp.exists(cfg.split_train) or not osp.exists(cfg.split_test):
            utils.split_meshes(cfg)
        if state is 'train':
            with open(cfg.split_train, 'rb') as f:
                self.shape_list = pickle.load(f)
        else:
            with open(cfg.split_test, 'rb') as f:
                self.shape_list = pickle.load(f)

        print('%s data num: %d'%(state, len(self.shape_list)))

    @staticmethod
    def normalize_adj(data):
        # print(data.shape)
        rowsum = np.array(data.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sparse.diags(r_inv)
        data = r_mat_inv.dot(data)
        return data

    @staticmethod
    def normalize_ft(data):
        # print(data.shape)
        centroid = np.mean(data, axis=0)
        data -= centroid
        furthest_disrance = np.max(np.sqrt(np.sum(abs(data)**2, axis=-1)))
        data /= furthest_disrance
        return data

    def __getitem__(self, index):
        shape = self.shape_list[index]
        num_verts, num_edge, num_faces, points, faces = utils.read_mesh(shape['mesh'])
        # print(shape['shape_name'])
        feature = np.array(points)
        label = shape['label']
        adj = np.eye(num_verts)
        for _i in range(num_faces):
            adj[faces[_i][1], faces[_i][2]], adj[faces[_i][2], faces[_i][1]] = 1, 1
            adj[faces[_i][1], faces[_i][3]], adj[faces[_i][3], faces[_i][1]] = 1, 1
            adj[faces[_i][2], faces[_i][3]], adj[faces[_i][3], faces[_i][2]] = 1, 1
        feature = self.normalize_ft(feature)
        adj = self.normalize_adj(adj)
        return torch.Tensor(feature), torch.Tensor(adj), torch.Tensor([label]).long()

    def __len__(self):
        return len(self.shape_list)


class PCData(Dataset):
    def __init__(self, cfg, state='train', shuffle=False, img_sz=227, ps_input_num=1024):
        super(PCData, self).__init__()
        if not osp.exists(cfg.split_train) or not osp.exists(cfg.split_test):
            utils.split_pc(cfg)

        if state=='train':
            with open(cfg.split_train, 'rb') as f:
                self.shape_list = pickle.load(f)
        else:
            with open(cfg.split_test, 'rb') as f:
                self.shape_list = pickle.load(f)

        if shuffle:
            random.shuffle(self.shape_list)

        print('%s data num: %d'%(state, len(self.shape_list)) )

        self.img_sz = img_sz
        self.cfg = cfg
        self.ps_input_num = ps_input_num

    @staticmethod
    def normalize_ft(data):
        # print(data.shape)
        centroid = np.mean(data, axis=0)
        data -= centroid
        furthest_disrance = np.max(np.sqrt(np.sum(abs(data)**2, axis=-1)))
        data /= furthest_disrance
        return data

    def __getitem__(self, idx):
        lbl = self.shape_list[idx]['label']
        pc = np.load(self.shape_list[idx]['pc'])[:self.ps_input_num].astype(np.float32)
        pc = np.expand_dims(pc.transpose(), axis=2)
        pc = self.normalize_ft(pc)
        return torch.from_numpy(pc), lbl

    def __len__(self):
        return len(self.shape_list)


if __name__ == '__main__':

    cfg = utils.config()
    m = MeshData(cfg, 'train')
    print(len(m))
    tmp = m[0]
    print(tmp[2])

