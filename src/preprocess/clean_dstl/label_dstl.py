import pandas as pd
import os
import numpy as np
import argparse
import json
import os
import sys
import shutil
import argparse
import numpy as np
from lxml import etree


def write_voc_file(fname, labels, coords, img_width, img_height):
    """
    Definition: Writes label into VOC (XML) format.
    Parameters: fname - full file path to label file
                labels - list of objects in file
                coords - list of position of objects in file
                img_width - width of image
                img_height - height of image
    Returns: annotation - XML tree for image file
    Credit: eweill/convert-datasets
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


def get_all_bounding(config):
    # get file
    ann_file = (os.getcwd() + config['dstl']['proc_data_rel'] +
                "annotations/annotations.csv")
    df = pd.read_csv(ann_file, header=None)
    df.columns = ['file', 'x_min', 'y_min', 'x_max', 'y_max', 'label_str']
    label_path = os.getcwd() + config['dstl']['proc_data_rel'] + "/labels"
    image_path = os.getcwd() + config['dstl']['proc_data_rel'] + "/images"
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
    print('Deleting {} files'.format(len(np.unique(files2delete))))
    # drop nans()
    df.dropna(axis=0, inplace=True)
    # set all the labels, after rerunning set just to trees
    label_dir = {'trees': 0, 'buildings': 1, 'animals': 2}
    df['label'] = df.label_str.apply(lambda x: label_dir[x]).astype('int')
    df['label'] = df['label'].astype('int')
    # get width and height
    w = df['x_max'] - df['x_min']
    h = df['y_max'] - df['y_min']
    width_image = config['dstl']['imag_w']
    height_image = config['dstl']['imag_w']
    # convert to x, y, w, h (scaled)
    x_center = (df['x_min'] + w / 2) / width_image
    y_center = (df['y_min'] + h / 2) / height_image
    df['w'] = w / width_image
    df['h'] = h / height_image
    df['x'] = x_center
    df['y'] = y_center
    return df, files2delete


def build_labels(df, files2delete, imag_w, imag_h, lab_format='voc'):
    # grab files
    files2use = pd.unique(df['file'])
    # clean all unusable files
    for f_name in files2delete:
        print('going to delete', f_name)

    # build labels for usable files
    for f_name in files2use:
        # convert data for file
        data_temp = df.loc[df['file'] == f_name]
        print(data_temp.columns)
        # split up path for conversion
        f_name_list = f_name.split('/')
        # get image name
        f_img = ('/'.join(f_name_list[:-2]) + '/images/'
                 + f_name_list[-1])
        # grab labels depending on format
        if lab_format == 'yolo':
            f_name_local_lab = f_name_list[-1][:-4]+'.txt'
            f_lab = ('/'.join(f_name_list[:-2]) + '/labels/'
                     + f_name_local_lab)
            with open(f_lab, 'w+') as f:
                for index, row in data_temp.iterrows():
                    str2dump = str([row['label'], row['x'],
                                    row['y'], row['w'], row['h']])
                    f.write(str2dump + '\n')
        elif lab_format == 'voc':
            f_name_local_lab = f_name_list[-1][:-4]+'.xml'
            f_lab = ('/'.join(f_name_list[:-2]) + '/labels/'
                     + f_name_local_lab)
            with open(f_lab, 'w+') as f:
                labels = data_temp.label_str
                coords = np.array(data_temp.loc[:, ['x_min', 'y_min', 'x_max', 'y_max']])
                print(coords)
                # convert to xml
                annot2dump = write_voc_file(f_img, labels, coords,
                                            imag_w, imag_h)
                et = etree.ElementTree(annot2dump)
                et.write(f_lab, pretty_print=True)
        else:
            error_str = ('Do not recognize label conversion format. ' +
                         'Must be voc or yolo')
            raise RuntimeError(error_str)
        print('going to move', f_name, 'to', f_img)
        break


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='config file')
    args = parser.parse_args()
    config_path = args.config
    print(os.getcwd())
    # load config
    with open(config_path) as config_buffer:
        # config = json.loads(config_buffer.read())
        config = json.load(config_buffer)

    # get all the bound box labels
    df, files2delete = get_all_bounding(config)

    # get all the bound box labels
    build_labels(df, files2delete, config['dstl']['imag_w'],
                 config['dstl']['imag_h'], lab_format='voc')
