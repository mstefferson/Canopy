import gdal
import os
import glob
from gdalconst import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pytictoc import TicToc
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import pickle
import streamlit as st


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
    peaks = np.zeros((len(where_peaks[0]), 2))
    peaks[:, 0] = where_peaks[0]
    peaks[:, 1] = where_peaks[1]
    return peaks


def get_satellite_subset(ds_all, r_start, c_start, r_del, c_del):
    '''
    Description:
        Returns a numpy data array of the rasted satellite
        image for all bands from a gdal object.
    Inputs:
        ds_all (osgeo.gdal.Dataset): Gdal data structure from opening a tif,
            ds_all = gldal.Open('...')
        r_start (int): Initial row pixel number of subset
        c_start (int): Initial column pixel number of subset
        r_del (int): Width of subset in pixels along rows
        c_del (int): Width of subset in pixels along columns
    Returns:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
    Updates:
        N/A
    Write to file:
        N/A
    '''
    channels = 3
    band_data = np.zeros((r_del, c_del, 4))
    data_sum = np.zeros((r_del, c_del))
    for index in np.arange(4):
        band_num = index + 1
        band = ds_all.GetRasterBand(band_num)
        data = band.ReadAsArray(r_start, c_start, r_del, c_del)
        # grab data
        band_data[:, :, index] = data
    # scale data
    band_data = band_data / np.max(band_data)
    return band_data


def get_tree_finder_image(band_data, drop_thres=0.05):
    '''
    Description:
        Returns a np.array that is strongely peaked
        where there are trees from the rastered data. The
        output uses the green and IR bands to get peaks along trees
    Inputs:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
        Threshold (float, optinal): The threshold peak value for
            counting it as a tree. 0.05 seems to work
    Returns:
        plant_data (np.array, size=[r_del, c_del]): A map that is strongly
            peaks where there are trees
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # All color bands peak at white light (which building are), get a average
    # to sub it out
    data_ave = ((band_data[:, :, 0] + band_data[:, :, 1] + band_data[:, :, 2])
                / 3)
    # Plants reflect green and IR so take a weighted average and subtract
    # out background. This weighted average is just based on messing
    # around with the data
    plant_data = (3*band_data[:, :, 1] + band_data[:, :, 3]) / 4 - data_ave
    plant_data[plant_data < drop_thres] = 0
    return plant_data


def plot_satellite_image(band_data, plant_data, tree_loc):
    '''
    Description:
        Plots the raster bands, the plant data, and tree locations
    Inputs:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
        plant_data (np.array, size=[r_del, c_del]): The rastered image
            data for all bands
    Returns:
        plant_data (np.array, size=[r_del, c_del]): A map that is strongly
            peaks where there are trees
        plant_data (np.array, size=[r_del, c_del]): A map that is strongly
            peaks where there are trees
    Updates:
        N/A
    Write to file:
        N/A
    '''
    st.title('Sample satellite images')
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


def find_peaks_for_subset(ds_all, x_start, y_start, x_del, y_del, plot_flag):
    '''
    Description:
        Grabs a data subset and finds all the trees. This is a wrapper
        for many functions that: grab subset, gets plant data, find trees,
        plots
    Inputs:
        ds_all (osgeo.gdal.Dataset): Gdal data structure from opening a tif,
            ds_all = gldal.Open('...')
        r_start (int): Initial row pixel number of subset
        c_start (int): Initial column pixel number of subset
        r_del (int): Width of subset in pixels along rows
        c_del (int): Width of subset in pixels along columns
        plot_flag (bool): Plot the subset
    Returns:
        plant_dict (dict): summary of tree locations and row/col indices
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # get the band data
    band_data = get_satellite_subset(ds_all, x_start, y_start, x_del, y_del)
    # get tree data
    plant_data = get_tree_finder_image(band_data)
    # get peaks
    tree_loc = detect_peaks(plant_data)
    # plot it
    if plot_flag:
        plot_satellite_image(band_data, plant_data, tree_loc)
    # store output
    plant_dict = {}
    leading_zeros = int(np.ceil(np.log10(ds_all.RasterXSize)))
    # build up string
    id_base = 'sat_'
    int_str_format = '{:0' + str(leading_zeros) + '}'
    # store the end of the range
    x_end = x_start + x_del
    y_end = y_start + y_del
    store_id = (id_base +
                int_str_format.format(x_start) + '_' +
                int_str_format.format(x_end) + '_' +
                int_str_format.format(y_start) + '_' +
                int_str_format.format(y_end) + '_'
                )
    # put peaks in a nice format
    peaks_zip_local = tree_loc
    peaks_zip_global = np.zeros(np.shape(peaks_zip_local))
    peaks_zip_global[:, 0] = peaks_zip_local[:, 0] + x_start
    peaks_zip_global[:, 1] = peaks_zip_local[:, 1] + y_start
    # store it
    plant_dict = {'store_id': store_id, 'x_start': x_start, 'x_end': x_end,
                  'y_start': y_start, 'y_end': y_end,
                  'trees_local': peaks_zip_local,
                  'trees_global': peaks_zip_global}
    return plant_dict


def main(sat_file, plot_flag):
    '''
    Description:
        Loops over an entire satellite image, divides it into subset,
        and finds trees for each subset. Writes a list of all
        tree locations (in pixels) to files as a pkl
    Inputs:
        sat_file (str): Path to satellite tif file
        plot_flag (bool): Plot the subset
    Returns:
        None
    Updates:
        N/A
    Write to file:
        'tree_coords.pkl': pickle file of a list of tree coordinates in pixels
    '''
    cwd = os.getcwd()
    # st.write('In directory:' + cwd)
    ds_all = gdal.Open(sat_file, GA_ReadOnly)
    # get raster bands for a subset
    x_start = ds_all.RasterXSize // 2
    y_start = ds_all.RasterYSize // 2
    x_del = 1000
    y_del = 1000
    x_end = x_start + 2 * x_del
    y_end = y_start + 2 * y_del
    counter = 0
    tree_coords = []
    # loop over subsets
    for xs in np.arange(x_start, x_end, x_del):
        for ys in np.arange(y_start, y_end, y_del):
            tree_dict = find_peaks_for_subset(ds_all, xs, ys,
                                              x_del, y_del, plot_flag)
            # store just the global tree coordinates
            if counter == 0:
                tree_coords = tree_dict['trees_global']
            else:
                tree_coords = np.append(tree_coords, tree_dict['trees_global'],
                                        axis=0)
            counter += 1
    # dump it
    pickle.dump(tree_coords, open('tree_coords.pkl', 'wb'))


if __name__ == '__main__':
    '''
    Description:
        Takes in commandline arguements and self main()
    Example call:
        python src/satellite_analyze data/raw/athens_satellite.tif --plot=true
    '''
    # start timer
    t = TicToc()
    t.tic()
    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('sat_file', type=str, help='path to satellite tif')
    parser.add_argument('--plot', type=bool, default=False, help='plot flag')
    args = parser.parse_args()
    # run main
    main(args.sat_file, args.plot)
    t.toc()
