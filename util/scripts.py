import os

import numpy as np
from PIL import Image


def get_ihd_training_data(dname, dataset_root, is_train=True):
    image_path = []
    dataset_list = ['HAdobe5k', 'HCOCO', 'Hday2night', 'HFlickr']
    if dname == 'ihd':
        for dataset in dataset_list:
            dataset_dir = os.path.join(dataset_root, dataset)
            assert os.path.isdir(dataset_dir), "ERROR: Make sure {} is in IHD Dataset directory".format(dataset)
            train_file = os.path.join(dataset_dir, dataset + ('_train.txt' if is_train else '_test.txt'))
            with open(train_file, 'r') as f:
                for line in f.readlines():
                    image_path.append(os.path.join(dataset_dir, 'composite_images', line.rstrip()))
    else:
        dataset_dir = os.path.join(dataset_root, dname)
        assert os.path.isdir(dataset_dir), "ERROR: Make sure {} is in IHD Dataset directory".format(dname)
        train_file = os.path.join(dataset_dir, dname + ('_train.txt' if is_train else '_test.txt'))
        with open(train_file, 'r') as f:
            for line in f.readlines():
                image_path.append(os.path.join(dataset_dir, 'composite_images', line.rstrip()))
    return image_path


def crop_foreground(composite, mask):
    composite, mask = np.array(composite), np.array(mask)
    top, bottom, left, right = mask.shape[0], 0, mask.shape[1], 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                top, bottom = min(top, i), max(bottom, i)
                left, right = min(left, j), max(right, j)

    fg = composite[top:bottom + 1, left:right + 1, :]
    return Image.fromarray(fg)
