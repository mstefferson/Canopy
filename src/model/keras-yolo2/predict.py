#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import json
from src.models.keras-yolo2.preprocessing import parse_annotation
from src.models.keras-yolo2.utils import draw_boxes
from src.models.keras-yolo2.frontend import YOLO

'''
https://github.com/experiencor/keras-yolo2/
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, config['model']['labels'])

    print(len(boxes), 'boxes are found')

    filename = image_path[:-4] + '_detected' + image_path[-4:]
    f = open('results.txt', 'w+')
    for box in boxes:
        array2write = [box.label, box.xmin, box.xmax, box.ymin, box.ymax,
                       box.c]
        f.write(str(array2write) + '\n')
    cv2.imwrite(filename, image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
