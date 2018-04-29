import sys
import os.path as osp
import pickle
import glob
# sys.path.append('../')
import utils


def split_meshes():
    cfg = utils.config()
    train_meshes = get_file_names('train', cfg.data_simply_mesh_root)
    test_meshes = get_file_names('test', cfg.data_simply_mesh_root)

    print('train number: %d'%len(train_meshes))
    print('test number: %d'%len(test_meshes))

    with open(cfg.split_train, 'wb') as f:
        pickle.dump(train_meshes, f)
    with open(cfg.split_test, 'wb') as f:
        pickle.dump(test_meshes, f)

    print('split done!')


def get_file_names(state, root):
    file_names = []
    class_dirs = glob.glob(osp.join(root, '*'))
    class_dirs = sorted(class_dirs)
    class_dirs = [class_dir for class_dir in class_dirs if osp.isdir(class_dir)]
    for label, class_dir in enumerate(class_dirs):
        class_name = osp.split(class_dir)[1]

        meshes = glob.glob(osp.join(class_dir, state, '*.off'))
        meshes = sorted(meshes)
        for mesh in meshes:
            shape_name = osp.split(mesh)[1].split('.')[0]
            file_names.append({'label': label,
                               'label_name': class_name,
                               'mesh': mesh,
                               'shape_name': shape_name})
    return file_names
