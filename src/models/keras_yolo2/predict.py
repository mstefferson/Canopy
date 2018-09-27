#! /usr/bin/env python

import argparse
import os
import sys
import cv2
import numpy as np
import json
sys.path.append(os.getcwd())
from src.models.keras_yolo2.preprocessing import parse_annotation
from src.models.keras_yolo2.utils import draw_boxes
from src.models.keras_yolo2.frontend import YOLO

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
    help='path to an image')

argparser.add_argument(
    '-b',
    '--bound',
    help='write bounding boxes to file')

argparser.add_argument(
    '-d',
    '--detect',
    help='save detect file')


def build_model(config, weights_path):

    # build model
    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    # load weights
    yolo.load_weights(weights_path)
    return yolo

def predict_bounding_box(yolo, image, labels):
    # predict
    bboxes = yolo.predict(image)
    # store bounding boxes in a more usable format
    box_list = []
    for box in bboxes:
        x_center = (box.xmax + box.xmin) / 2
        width = (box.xmax - box.xmin) / 2
        y_center = (box.ymax + box.ymin) / 2
        height = (box.ymax - box.ymin) / 2
        box_list.append([box.label, image.shape[1], image.shape[0], 
                          x_center, y_center, width, height, box.c])

    return box_list, bboxes

def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input
    save_detect = args.detect
    write_file = args.bound
    # build config
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    # build model
    yolo = build_model(config, weights_path)
    # predict
    # load image and predict bounding box
    image = cv2.imread(image_path)
    # get boxes 
    box_list, bboxes = predict_bounding_box(yolo, image, config['model']['labels'])
    if write_file:
        f = open('results.txt', 'w+')
        for box in box_list:
            f.write(str(box) + '\n')
    if  save_detect:
        image = draw_boxes(image, bboxes, config['model']['labels'])
        filename = image_path[:-4] + '_detected' + image_path[-4:]
        cv2.imwrite(filename, image)
        

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
