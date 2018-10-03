import numpy as np
import os
import glob


def remove_singles(path1, path2):
    print(path1)
    print(path2)
    list1 = glob.glob(path1 + '/*')
    list2 = glob.glob(path2 + '/*')
    print('1:', len(list1), '2:', len(list2))
    strip1 = [a_str.split('/')[-1][:-4] for a_str in list1]
    strip2 = [a_str.split('/')[-1][:-4] for a_str in list2]
    print(strip1[:10])
    print(strip2[:10])
    intersect = set(strip1).intersection(set(strip2))
    remove1 = set(strip1) - set(strip2)
    remove2 = set(strip2) - set(strip1)
    print('intersect:', len(intersect))
    print('r1:', len(remove1), 'r2:', len(remove2))
    for f in remove1:
        fremove = path1 + '/' + f + '.png'
        os.remove(fremove)
    for f in remove2:
        fremove = path2 + '/' + f + '.xml'
        os.remove(fremove)


if __name__ == '__main__':
    print('Training')
    path_images = os.getcwd() + '/data/processed/dstl/train/images'
    path_label = os.getcwd() + '/data/processed/dstl/train/labels'
    remove_singles(path_images, path_label)
    print('Valid')
    path_images = os.getcwd() + '/data/processed/dstl/valid/images'
    path_label = os.getcwd() + '/data/processed/dstl/valid/labels'
    remove_singles(path_images, path_label)
