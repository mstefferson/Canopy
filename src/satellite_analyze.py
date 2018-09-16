import gdal
import os
import glob
from gdalconst import *
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background
    # from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background,
                                       structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    peak_mask = local_max ^ eroded_background
    peaks = np.where(peak_mask)
    return peaks


def get_satellite_subset(ds_all, x_start, y_start, x_del, y_del):
    channels = 3
    band_data = np.zeros((x_del, y_del, 4))
    data_sum = np.zeros((x_del, y_del))
    for index in np.arange(4):
        band_num = index + 1
        band = ds_all.GetRasterBand(band_num)
        data = band.ReadAsArray(x_start, y_start, x_del, y_del)
        # grab data
        band_data[:, :, index] = data
    # scale data
    band_data = band_data / np.max(band_data)
    return band_data


def get_tree_finder_image(data_all, drop_thres=0.05):
    data_ave = (data_all[:, :, 0] + data_all[:, :, 1] + data_all[:, :, 2]) / 3
    plant_data = (3*data_all[:, :, 1] + data_all[:, :, 3]) / 4 - data_ave
    plant_data[plant_data < drop_thres] = 0
    return plant_data


def plot_plants(band_data, plant_data, tree_loc):
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs = np.reshape(axs, [6, ])
    color_scheme = ['red', 'green', 'blue',  'IR']
    for index in np.arange(4):
        # plot it
        plt.subplot(2, 3, index+1)
        plt.title(color_scheme[index])
        imgplot = plt.imshow(band_data[:, :, index])
    # plot it
    plt.subplot(2, 3, 5)
    plt.title('All colors')
    imgplot = plt.imshow(band_data[:, :, :3])
    # plot it
    plt.subplot(2, 3, 6)
    imgplot = plt.imshow(plant_data)
    # plot trees (x and y are switched)
    plt.scatter(tree_loc[1], tree_loc[0], color='r')
    plt.title('Plant detect')
    # plt.show()
    st.pyplot()


def find_tree_peaks(plant_data):
    print('Not written')


def main():
    cwd = os.getcwd()
    st.write('In directory:' + cwd)
    filename = 'data/raw/athens_satellite.tif'
    ds_all = gdal.Open(filename, GA_ReadOnly)
    # get raster bands for a subset
    x_start = ds_all.RasterXSize // 2 + 100
    y_start = ds_all.RasterYSize // 2 - 100
    x_del = 1000
    y_del = 1000
    band_data = get_satellite_subset(ds_all, x_start, y_start, x_del, y_del)
    # get tree data
    plant_data = get_tree_finder_image(band_data)
    # get peaks
    tree_loc = detect_peaks(plant_data)
    # plot it
    plot_plants(band_data, plant_data, tree_loc)


if __name__ == '__main__':
    main()
