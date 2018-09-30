import pandas as pd
import os
import numpy as np
import argparse
import json
import sys
import shutil
import argparse
import numpy as np
from lxml import etree


def write_voc_file(fname, labels, coords, img_width, img_height):
    """
    Writes label into VOC (XML) format.

    Args:
        fname - full file path to label file
        labels - list of objects in file
        coords - list of position of objects in file
        img_width - width of image
        img_height - height of image
    Returns: 
        annotation - XML tree for image file
    Updates:
        N/A
    Writes to file:
        N/A
    Credit: 
        eweill/convert-datasets
    """
    annotation = etree.Element('annotation')
    filename = etree.Element('filename')
    f = fname.split("/")
    filename.text = f[-1]
    annotation.append(filename)
    folder = etree.Element('folder')
    folder.text = "/".join(f[:-1])
    annotation.append(folder)
    for i in range(len(coords)):
        object = etree.Element('object')
        annotation.append(object)
        name = etree.Element('name')
        name.text = labels[i]
        object.append(name)
        bndbox = etree.Element('bndbox')
        object.append(bndbox)
        xmax = etree.Element('xmax')
        xmax.text = str(coords[i][2])
        bndbox.append(xmax)
        xmin = etree.Element('xmin')
        xmin.text = str(coords[i][0])
        bndbox.append(xmin)
        ymax = etree.Element('ymax')
        ymax.text = str(coords[i][3])
        bndbox.append(ymax)
        ymin = etree.Element('ymin')
        ymin.text = str(coords[i][1])
        bndbox.append(ymin)
        difficult = etree.Element('difficult')
        difficult.text = '0'
        object.append(difficult)
        occluded = etree.Element('occluded')
        occluded.text = '0'
        object.append(occluded)
        pose = etree.Element('pose')
        pose.text = 'Unspecified'
        object.append(pose)
        truncated = etree.Element('truncated')
        truncated.text = '1'
        object.append(truncated)
    img_size = etree.Element('size')
    annotation.append(img_size)
    depth = etree.Element('depth')
    depth.text = '3'
    img_size.append(depth)
    height = etree.Element('height')
    height.text = str(img_height)
    img_size.append(height)
    width = etree.Element('width')
    width.text = str(img_width)
    img_size.append(width)
    return annotation


def get_all_bounding(process_path, imag_w, imag_h):
    '''
    Gets all of the bounding box informations for the annotations
        file built from clean_dstl

    Args:
        process_path (str): Path to processed data (parent of chopped images 
            and annotations file). This should be the processed data path 
            in config
        imag_w (int): Chopped image width
        imag_h (int): Chopped image height
    Returns:
        df (pd.Dataframe): Dataframe with the bounding box info
            for all of the chopped images (from clean_dstl)
        files2delete (list): List all the unlabeled chopped images
    Updates:
        N/A
    Writes to file:
        N/A
    '''
    # get file
    ann_file = (os.getcwd() + process_path +
                "annotations.csv")
    df = pd.read_csv(ann_file, header=None)
    df.columns = ['file', 'x_min', 'y_min', 'x_max', 'y_max', 'label_str']
    label_path = os.getcwd() + process_path + "/labels"
    image_path = os.getcwd() + process_path + "/images"
    # make dir
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    # make dir
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    df = pd.read_csv(ann_file, header=None)
    df.columns = ['file', 'x_min', 'y_min', 'x_max', 'y_max', 'label_str']
    # find and remove all nans
    files2delete = df.file.loc[df.isnull().any(axis=1)]
    # drop nans()
    df.dropna(axis=0, inplace=True)
    # set all the labels, after rerunning set just to trees
    label_dir = {'trees': 0, 'buildings': 1, 'animals': 2}
    df['label'] = df.label_str.apply(lambda x: label_dir[x]).astype('int')
    df['label'] = df['label'].astype('int')
    # get width and height
    w = df['x_max'] - df['x_min']
    h = df['y_max'] - df['y_min']
    width_image = imag_w
    height_image = imag_h
    # convert to x, y, w, h (scaled)
    x_center = (df['x_min'] + w / 2) / width_image
    y_center = (df['y_min'] + h / 2) / height_image
    df['w'] = w / width_image
    df['h'] = h / height_image
    df['x'] = x_center
    df['y'] = y_center
    return df, files2delete


def build_labels(df, files2delete, imag_w, imag_h, lab_format='voc'):
    '''
    Given a df, builds image and label directories for the images
        with labels and deletes the unused ones.
    
    Args:
        df (pd.Dataframe): Dataframe of the annotations from clean_dstl.
            Contains path info and labels for each images in dstl.
        files2delete (list of strs): List of the full path of unlabeled
            data that will be deleted
        imag_w (int): Chopped image width
        imag_h (int): Chopped image height
        lab_format (str): Label format. Accepts 'voc'/'yolo'
    Returns:
        N/A
    Updates:
        N/A
    Writes to file:
        Deletes unlabeled images and moves labeled images from
        /base/path/data/processed/chopped_images to 
        /base/path/data/processed/(images, labels)
    '''
    # clean all unusable files
    print('Deleting {} files'.format(len(np.unique(files2delete))))
    for f_name in files2delete:
        if os.path.exists(f_name):
            os.remove(f_name)
    print('Deleting {} files'.format(len(np.unique(files2delete))))
    if len(df) > 0:
        # get base directory
        fullpath = df.iloc[0, 0]
        base_dir = '/'.join(fullpath.split('/')[:-2])
        chopped_dir = '/'.join(fullpath.split('/')[:-1])
        # build outputs
        for a_dir in [base_dir + '/images/', base_dir + '/labels/']:
            if not os.path.exists(a_dir):
                os.makedirs(a_dir)
        # grab files
        files2use = pd.unique(df['file'])
        # build labels for usable files
        print('Found {} labeled files'.format(len(np.unique(files2use))))
        for f_name in files2use:
            # convert data for file
            data_temp = df.loc[df['file'] == f_name]
            # split up path for conversion
            f_name_list = f_name.split('/')
            # get image name
            f_img = base_dir + '/images/' + f_name_list[-1]
            # grab labels depending on format
            if lab_format == 'yolo':
                f_name_local_lab = f_name_list[-1][:-4]+'.txt'
                f_lab = base_dir + '/labels/' + f_name_local_lab
                with open(f_lab, 'w+') as f:
                    for index, row in data_temp.iterrows():
                        str2dump = str([row['label'], row['x'],
                                        row['y'], row['w'], row['h']])
                        f.write(str2dump + '\n')
            elif lab_format == 'voc':
                f_name_local_lab = f_name_list[-1][:-4]+'.xml'
                f_lab = base_dir + '/labels/' + f_name_local_lab
                with open(f_lab, 'w+') as f:
                    labels = list(data_temp.label_str)
                    coords = np.array(
                        data_temp.loc[:, ['x_min', 'y_min', 'x_max', 'y_max']])
                    # convert to xml
                    annot2dump = write_voc_file(f_img, labels, coords,
                                                imag_w, imag_h)
                    et = etree.ElementTree(annot2dump)
                    et.write(f_lab, pretty_print=True)
            else:
                error_str = ('Do not recognize label conversion format. ' +
                             'Must be voc or yolo')
                raise RuntimeError(error_str)
            os.rename(f_name, f_img)
        # remove chopped_files directory
        if not os.listdir(chopped_dir):
            print('Chopped empty, deleting')
            os.rmdir(chopped_dir)
        else:
            print('Chopped not empty, not deleting')
    else:
        print('No labeled data')


def build_val_train(path2data, val_size=0.3):
    '''
    Splits the images/labels from an input path to a train and validation
    set

    Args:
        path2data (str): base path to the images and labels (e.g.,
            /path2data/images, /path2data/labels
        val_size (float, optional): validation set fraction
    Returns:
        N/A
    Updates:
        N/A
    Writes to file:
        Removes /path2data/(images, labels) and writes 
            /path2data/(train, val)/(images, labels)
    '''
    # set up paths
    image_path = path2data + 'images/'
    label_path = path2data + 'labels/'
    if os.path.exists(image_path) and os.path.exists(label_path):
        image_path_train = path2data + 'train/images/'
        label_path_train = path2data + 'train/labels/'
        image_path_val = path2data + 'val/images/'
        label_path_val = path2data + 'val/labels/'
        # make dirs
        all_dirs =  [image_path_train, label_path_train,
                     image_path_val, label_path_val]
        for a_dir in all_dirs:
            if not os.path.exists(a_dir):
                os.makedirs(a_dir)
        # grab all files
        all_images = [f for f in os.listdir(image_path)
                      if os.path.isfile(os.path.join(image_path, f))]
        all_labels = [f for f in os.listdir(label_path)
                      if os.path.isfile(os.path.join(label_path, f))]
        # verify the lengths are the same (should compare sets)
        if len(all_images) != len(all_labels):
            raise RuntimeError('Images/label mismatch!')
        # get all the files
        num_files = len(all_images)
        num_valid = int(num_files * 0.3)
        num_train = num_files - num_valid
        # get all indices and shuffle them
        all_ids = np.arange(num_files)
        np.random.shuffle(all_ids)
        train_ids = all_ids[1:num_train]
        val_ids = all_ids[num_train:]
        # move all files
        for ind in all_ids:
            # train
            if ind <= num_train:
                os.rename(image_path+all_images[ind],
                          image_path_train+all_images[ind])
                os.rename(label_path+all_labels[ind],
                          label_path_train+all_labels[ind])
            # val
            else:
                os.rename(image_path+all_images[ind],
                          image_path_val+all_images[ind])
                os.rename(label_path+all_labels[ind],
                          label_path_val+all_labels[ind])
        # remove empty directories
        for a_dir in [image_path, label_path]:
            if not os.listdir(a_dir):
                print(str(a_dir) + ' empty, deleting')
                os.rmdir(a_dir)
            else:
                print(str(a_dir) + ' not empty, not deleting')
    else:
        print('No data to move to train/val')


def main(config):
    '''
    After cleaning the dstl data, i.e. building annotations and chopping up
        the files by running clean_dstl, this function will build a nice
        labeled traning and validation set with the appropriate labeling
        format
    Args:
        config (dist): loaded dstl config json. Contains image and path info
    Returns:
        N/A
    Update:
        N/A
    Writes to file:
        Removes /path2data/chopped_images and writes
            /path2data/(train, val)/(images, labels)
    '''
    # get all the bound box labels
    df, files2delete = get_all_bounding(config['dstl']['proc_data_rel'],
                                        config['dstl']['imag_w'],
                                        config['dstl']['imag_h'])
    print('Got all bounding boxes')
    # get all the bound box labels
    build_labels(df, files2delete, config['dstl']['imag_w'],
                 config['dstl']['imag_h'],
                 lab_format=config['dstl']['label_format'])
    print('Built all labels')
    # build val/train
    path2data = os.getcwd() + config['dstl']['proc_data_rel']
    build_val_train(path2data, val_size=0.3)
    print('Move to train/val')


if __name__ == '__main__':
    '''
    Executeable:
    python3 src/preprocess/clean_dstl/label_dstl.py /
        configs/config_dstl.json
    '''
    # parse args

    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path to config file')
    args = parser.parse_args()
    config_path = args.config
    print(os.getcwd())
    # load config
    with open(config_path) as config_buffer:
        # config = json.loads(config_buffer.read())
        config = json.load(config_buffer)
    # run main
    main(config)
