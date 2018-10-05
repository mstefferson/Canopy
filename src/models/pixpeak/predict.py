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


def detect_peaks(array_with_peaks):
    """
    Description:
    Takes a 2D array and detects all peaks using the local maximum filter.
    Code adapted from:
    https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    Inputs:
        array_with_peaks (np.array): 2d array to find peaks of
    Returns:
        peaks (np.array, size=[num_trees, 2]): the row/column
            coordinates for all tree found
    Updates:
        N/A
    Write to file:
        N/A
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(array_with_peaks,
                               footprint=neighborhood) == array_with_peaks
    # remove background from image
    # we create the mask of the background
    background = (array_with_peaks == 0)
    # Erode background and border
    eroded_background = binary_erosion(background,
                                       structure=neighborhood, border_value=1)
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    peak_mask = local_max ^ eroded_background
    # grab the peaks
    where_peaks = np.where(peak_mask)
    # put them in np array
    num_peaks = len(where_peaks[0])
    peaks = np.zeros((num_peaks, 2))
    peaks[:, 0] = where_peaks[0]
    peaks[:, 1] = where_peaks[1]
    return peaks


def get_satellite_subset(ds_all, r_start, r_end, c_start, c_end,
                         norm=None):
    '''
    Description:
        Returns a numpy data array of the rasted satellite
        image for all bands from a gdal object.
    Args:
        ds_all (rasterio.io.DatasetReader): Gdal data structure from opening a
            tif, ds_all = rasterio.open('...')
        r_start (int): Initial row pixel number of subset
        c_start (int): Initial column pixel number of subset
        r_end (int): Initial row pixel number of subset
        c_end (int): Final row pixel number of subset
        norm (float): normalization value (max(array) <= norm)
    Returns:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # convert to int
    r_start = int(r_start)
    r_end = int(r_end)
    c_start = int(c_start)
    c_end = int(c_end)
    # set 4 channels (r,g,b,IR)
    channels = 4
    # calculate width
    r_del = r_end - r_start
    c_del = c_end - c_start
    # initialize
    band_data = np.zeros((r_del, c_del, channels))
    data_sum = np.zeros((r_del, c_del))
    # get end point
    r_end = r_start + r_del
    c_end = c_start + c_del
    for index in range(channels):
        band_num = index + 1
        data = ds_all.read(band_num,
                           window=((r_start, r_end), (c_start, c_end)))
        # grab data
        band_data[:, :, index] = data
    # scale data
    if norm:
        tifmax = 65535.
        band_data = norm / tifmax * band_data
    return band_data


def get_tree_finder_image(band_data, drop_thres=0.05):
    '''
    Description:
        Returns a np.array that is strongely peaked
        where there are trees from the rastered data. The
        output uses the green and IR bands to get peaks along trees
    Args:
        band_data (np.array, size=[r_del, c_del, 3/4]): The rastered image
            data for all bands
        Threshold (float, optional): The threshold peak value for
            counting it as a tree. 0.05 seems to work
    Returns:
        plant_data (np.array, size=[r_del, c_del]): A map that is strongly
            peaks where there are trees
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # 4 bands: r, g, b, IR
    data_ave = ((band_data[:, :, 0] + band_data[:, :, 1] +
                 band_data[:, :, 2]) / 3)
    # Plants reflect green and IR so take a weighted average and subtract
    # out background. This weighted average is just based on messing
    # around with the data
    plant_data = (3*band_data[:, :, 1] + band_data[:, :, 3]) / 4 - data_ave
    plant_data[plant_data < drop_thres] = 0
    return plant_data


def predict_bounding_box(image):
    # predict to get boxes
    peaks = detect_peaks(image)
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


def draw_boxes(image, box_df):
    # turn pixels of tree red
    for _, row in box_df.iterrows():
        row_tree = row['y'] * row['imag_h']
        col_tree = row['x'] * row['imag_w']
        min_r_tree = np.max([0, row_tree - 3]).astype(int)
        max_r_tree = np.min([row['imag_h'], row_tree + 3]).astype(int)
        min_c_tree = np.max([0, col_tree - 3]).astype(int)
        max_c_tree = np.min([row['imag_w'], col_tree + 3]).astype(int)
        image[min_r_tree:max_r_tree, min_c_tree:max_c_tree, 0] = 0
        image[min_r_tree:max_r_tree, min_c_tree:max_c_tree, 1] = 0
        image[min_r_tree:max_r_tree, min_c_tree:max_c_tree, 2] = 255
    return image


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
        band_data = get_satellite_subset(
            sat_data, r_org, r_org+imag_w, c_org, c_org+imag_w, norm=1)
        plant_data = get_tree_finder_image(band_data, drop_thres=0.05)
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
            image = draw_boxes(image, box_df)
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
    main(config)
