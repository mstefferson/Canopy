import argparse
import glob
import os
import sys
import cv2
import numpy as np
import json
import pandas as pd
import sys
import glob
import os
import predict_pixpeak
import predict_yolo


def main(args):
    '''
    Predict the bounding boxes for a single image
    Args:
        args (argparser.parse_args object): argument object with attibutes:
            args.conf: config file
            args.weights: weight to trained yolo2 model
            args.input: image file to predict on
            args.bound: boolean to write bounding boxes to file
            args.detect: boolean to draw bounding boxes image
    Returns:
        N/A
    Updates:
        N/A
    Writes to file:
        If flags are set, writes detected image (image with bounding boxes)
            and the bounding box locations to file. The outputs are located
            in directories with the same base path as the images,
            /base/path/images
    '''
    # set configs
    config_path = args.conf
    # build config
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    # run predictions based on  model
    if config['model'] == 'pixpeak':
        print('Running pixpeak')
        predict_pixpeak.main(config)
    elif config['model'] == 'yolo2':
        print('Running yolo')
        predict_yolo.main(config)
    else:
        print('Do not recognize model')


if __name__ == '__main__':
    '''
    Executeable:
    python src/models/predict.py \
    -c configs/config.json \
    Credit: Code adapted from experiencor/keras-yolo2
    '''
    # set-up arg parsing
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
    args = argparser.parse_args()
    # run main
    main(args)
