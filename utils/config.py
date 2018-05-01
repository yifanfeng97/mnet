# import sys
# sys.path.append('../')
import os
import os.path as osp
import configparser


class config(object):
    def __init__(self, cfg_file='../config/config.cfg'):
        super(config, self).__init__()
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)

        # default
        self.data_root = cfg.get('DEFAULT', 'data_root')
        self.data_pc_root = cfg.get('DEFAULT', 'data_pc_root')
        self.data_mesh_root = cfg.get('DEFAULT', 'data_mesh_root')
        self.data_std_mesh_root = cfg.get('DEFAULT', 'data_std_mesh_root')
        self.data_simply_mesh_root = cfg.get('DEFAULT', 'data_simply_mesh_root')
        self.result_root = cfg.get('DEFAULT', 'result_root')
        self.class_num = cfg.getint('DEFAULT', 'class_num')
        self.vis_pc = cfg.getboolean('DEFAULT', 'vis_pc')
        self.model_type = cfg.get('DEFAULT', 'model_type')

        # train
        self.cuda = cfg.getboolean('TRAIN', 'cuda')
        self.resume_train = cfg.getboolean('TRAIN', 'resume_train')

        self.result_sub_folder = cfg.get('TRAIN', 'result_sub_folder')
        self.ckpt_folder = cfg.get('TRAIN', 'ckpt_folder')
        self.split_folder = cfg.get('TRAIN', 'split_folder')

        self.split_train = cfg.get('TRAIN', 'split_train')
        self.split_test = cfg.get('TRAIN', 'split_test')
        self.ckpt_model = cfg.get('TRAIN', 'ckpt_model')
        self.ckpt_optim = cfg.get('TRAIN', 'ckpt_optim')
        self.log_dir = cfg.get('TRAIN', 'log_dir')

        self.gpu = cfg.get('TRAIN', 'gpu')
        self.batch_size = cfg.getint('TRAIN', 'batch_size')
        self.workers = cfg.getint('TRAIN', 'workers')
        self.max_epoch = cfg.getint('TRAIN', 'max_epoch')
        self.n_neighbor = cfg.getint('TRAIN', 'n_neighbor')
        self.lr = cfg.getfloat('TRAIN', 'lr')
        self.weight_decay = cfg.getfloat('TRAIN', 'weight_decay')
        self.optimizer = cfg.get('TRAIN', 'optimizer')
        self.decay_step = cfg.getint('TRAIN', 'decay_step')
        self.decay_rate = cfg.getfloat('TRAIN', 'decay_rate')
        self.print_freq = cfg.getint('TRAIN', 'print_freq')

        self.check_dirs()

    def check_dir(self, folder):
        if not osp.exists(folder):
            os.mkdir(folder)

    def check_dirs(self):
        self.check_dir(self.data_pc_root)
        self.check_dir(self.data_mesh_root)
        self.check_dir(self.data_std_mesh_root)
        self.check_dir(self.data_simply_mesh_root)
        self.check_dir(self.result_root)
        self.check_dir(self.result_sub_folder)
        self.check_dir(self.ckpt_folder)
        self.check_dir(self.split_folder)
        self.check_dir(self.log_dir)
