import os
import sys
import shutil
import glob
import argparse
import numpy as np
from PIL import Image
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


if __name__ == '__main__':
    annotation = write_voc_file('blah',
                                ['trees', 'poop'],
                                [[0.1, 0.11, 0.2, 0.21],
                                 [0.3, 0.31, 0.4, 0.41]],
                                300,
                                300)
