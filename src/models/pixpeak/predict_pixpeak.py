from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import argparse
import json
import os
import glob
import cv2
import pandas as pd
import rasterio
import utils_pixpeak


def predict_bounding_box(image):
    '''
    Returns the pixpeak bounding box prediction given an image

    Args:
        image (np.array, shape=[imag_w, imag_h]): image to find
            peaks in

    Returns:
        box_df (pandas.df): Pandas dataframe contain info about the
            bounding boxes

    Updates:
        N/A

    Write to file:
        N/A
    '''
    # predict to get boxes
    peaks = utils_pixpeak.detect_peaks(image)
    # store it in a dataframe
    df_cols = ['label', 'imag_w', 'imag_h',
               'x', 'y', 'w', 'h', 'conf']
    box_df = pd.DataFrame(index=np.arange(len(peaks)),
                          columns=df_cols)
    for row in np.arange(len(peaks)):
        x_center = peaks[row, 1] / image.shape[1]
        width = 0
        y_center = peaks[row, 0] / image.shape[0]
        height = 0
        label = 0
        conf = 1
        box_df.loc[row, df_cols] = [label, image.shape[1], image.shape[0],
                                    x_center, y_center, width, height, conf]
    return box_df


def main(config):
    '''
    Predicts the bounding boxes for a single image. The code
        extracts the row and column from the file name and then
        grabs the corresponding 4 channel image from the tif file

    Args:
        config (dict): Config dictionary

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
    save_detect = config["save_detect"]
    save_bb = config["save_bb"]
    # get all the files you want to predict on
    pred_path = config["image_path"]
    # pred path can be a folder or image. Grab files accordingly
    if os.path.isfile(pred_path):
        files_2_pred = [pred_path]
    else:
        files_2_pred = glob.glob(pred_path + '*')
    print('Predicting objects on files', files_2_pred)
    # get sat data
    sat_data = rasterio.open(config["sat_tif"])
    # loop over all files
    for image_path in files_2_pred:
        # get r,c from file
        file_id = image_path.split('/')[-1]
        (r_org, c_org) = map(int, file_id[:-4].split('_')[-2:])
        # grab image
        image = cv2.imread(image_path)
        imag_w = image.shape[1]
        imag_h = image.shape[0]
        # get satellite info
        band_data = utils_pixpeak.get_satellite_subset(
            sat_data, r_org, r_org+imag_w, c_org, c_org+imag_w, norm=1)
        plant_data = utils_pixpeak.get_tree_finder_image(
            band_data, drop_thres=0.05)
        # build bounding boxes
        box_df = predict_bounding_box(plant_data)
        # get base directory for writing files
        path_info_list = image_path.split('/')
        base_dir = os.getcwd()
        file_name = path_info_list[-1]
        file_id = file_name[:-4]
        if save_bb:
            # build file names and directories
            result_dir = config["bb_folder"]
            path2write = base_dir + '/' + result_dir
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            filename = path2write + file_id + '.csv'
            box_df.to_csv(filename, index=False)
        if save_detect:
            # build file names and directories
            result_dir = config["detect_folder"]
            path2write = base_dir + '/' + result_dir
            filename = (path2write + file_id +
                        '_detected' + file_name[-4:])
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            image = utils_pixpeak.draw_boxes(image, box_df)
            cv2.imwrite(filename, image)


if __name__ == '__main__':
    '''
    Executeable:
    python src/models/pixpeak_predict.py -c configs/config.json
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
    main(config)
