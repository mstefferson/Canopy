#! /usr/bin/env python
import argparse
import glob
import os
import sys
import cv2
import numpy as np
import json
import pandas as pd
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_model(config, weights_path):
    '''
    Build the model
    Args:
        config (dict): configuration dictionary
        weights_path (str): path to pretrained weights
    Returns:
        yolo (keras model, YOLO class): keras model
    Updates:
        N/A
    Writes to file:
        N/A
    '''
    # build model
    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    # load weights
    print('Using weights', weights_path)
    yolo.load_weights(weights_path)
    return yolo


def predict_bounding_box(model, image, iou_threshold):
    '''
    Predicts the bounding boxes for a single image
    Args:
        yolo (keras model, YOLO class): keras model
        image (np.array): image to predict on
    Returns:
        box_list (list of floats): a list of the bounding box labels
        bboxes (list of BoundBox): a list of BoundBox class for each label
    Updates:
        N/A
    Writes to file:
        N/A
    '''
    # predict
    bboxes = model.predict(image, iou_threshold=iou_threshold)
    # store bounding boxes in a more usable format
    df_cols = ['label', 'imag_w', 'imag_h',
               'x', 'y', 'w', 'h', 'conf']
    box_df = pd.DataFrame(index=np.arange(len(bboxes)),
                          columns=df_cols)
    for (row, box) in enumerate(bboxes):
        x_center = (box.xmax + box.xmin) / 2
        width = (box.xmax - box.xmin) / 2
        y_center = (box.ymax + box.ymin) / 2
        height = (box.ymax - box.ymin) / 2
        label = box.label
        conf = box.c
        box_df.loc[row, df_cols] = [label, image.shape[1], image.shape[0],
                                    x_center, y_center, width, height, conf]
    return box_df, bboxes


def main(config):
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
    # get params
    weights_path = config["predict"]["weights"]
    save_detect = config["predict"]["save_detect"]
    save_bb = config["predict"]["save_bb"]
    # build model
    yolo_model = build_model(config, weights_path)
    # get all the files you want to predict on
    pred_path = config["predict"]["image_path"]
    # pred path can be a folder or image. Grab files accordingly
    if os.path.isfile(pred_path):
        files_2_pred = [pred_path]
    else:
        files_2_pred = glob.glob(pred_path + '*')
    print('Predicting objects on files', files_2_pred)
    # loop over all files
    for image_path in files_2_pred:
        # load image and predict bounding box
        image = cv2.imread(image_path)
        # predict to get boxes
        box_df, bboxes = predict_bounding_box(
            yolo_model, image, iou_threshold=config['model']['iou_threshold'])
        # get base directory for writing files
        path_info_list = image_path.split('/')
        base_dir = os.getcwd()
        file_name = path_info_list[-1]
        file_id = file_name[:-4]
        if save_bb:
            # build file names and directories
            result_dir = config["predict"]["bb_folder"]
            path2write = base_dir + '/' + result_dir
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            filename = path2write + file_id + '.csv'
            box_df.to_csv(filename, index=False)

        if save_detect:
            # build file names and directories
            result_dir = config["predict"]["detect_folder"]
            path2write = base_dir + '/' + result_dir
            filename = (path2write + file_id +
                        '_detected' + file_name[-4:])
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            image = draw_boxes(image, bboxes, config['model']['labels'])
            cv2.imwrite(filename, image)


if __name__ == '__main__':
    '''
    Executeable:
    python src/models/keras_yolo2/predict.py \
    -c configs/config_yolo.json \
    -w model_weights/full_yolo_tree.h5 \
    -b true \
    -d true \
    -i data/processed/test/images/image_07.jpg

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
    # set configs
    config_path = args.conf
    # build config
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    # run main
    main(configs)
