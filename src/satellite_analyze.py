import logging
import os
import glob
import numpy as np
import argparse
from pytictoc import TicToc
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import pickle
import geopandas
import pyproj
import rasterio


def get_latlon_tree_file(filename):
    # Read data using geopandas
    data = geopandas.read_file(filename)
    # map data to lat/long using projection type epsg:4326
    data = data.to_crs({"init": "epsg:4326"})
    # get lat from lon from bounds of point object
    lon = data['geometry'].apply(lambda x: x.bounds[0])
    lat = data['geometry'].apply(lambda x: x.bounds[1])
    # store it in numpy array
    lon_lat = np.zeros((len(lon), 2))
    lon_lat[:, 0] = lon
    lon_lat[:, 1] = lat
    return lon_lat


def get_xy_tree_file(filename):
    # Read data using geopandas
    data = geopandas.read_file(filename)
    # get lat from lon from bounds of point object
    x = data['geometry'].apply(lambda x: x.bounds[0])
    y = data['geometry'].apply(lambda x: x.bounds[1])
    # store it in numpy array
    xy = np.zeros((len(x), 2))
    xy[:, 0] = x
    xy[:, 1] = y
    return xy


def get_known_tree_all(transfunc=get_latlon_tree_file):
    # root = '/app/'
    root = '/Users/mike/Insight/Tree_Bot/'
    file1 = root + 'data/raw/known_trees/prasino_istorikou_kentou/dendra.shp'
    file2 = root + 'data/raw/known_trees/prasino_istorikou_kentou/glastra.shp'
    file3 = root + 'data/raw/known_trees/prasino_istorikou_kentou/pagkakia.shp'
    # get all the data
    data1 = transfunc(file1)
    data2 = transfunc(file2)
    data3 = transfunc(file3)
    # collect all data together
    data_all = np.append(data1, np.append(data2, data3, axis=0), axis=0)
    return data_all


def proj_latlon_2_rc(lon, lat, dataset):
    # input lat/lon
    in_proj = pyproj.Proj(init='epsg:4326')
    # output based on crs of tif/shp
    out_proj = pyproj.Proj(dataset.crs)
    # transform lat/lon to xy
    x, y = pyproj.transform(in_proj, out_proj, lon, lat)
    # convert rows and columns to xy map project
    (r, c) = rasterio.transform.rowcol(dataset.transform, x, y)
    return (int(r), int(c))


def proj_rc_2_latlon(r, c, dataset):
    # convert rows and columns to xy map project
    (x, y) = rasterio.transform.xy(dataset.transform, r, c)
    # input based on crs of tif/shp
    in_proj = pyproj.Proj(dataset.crs)
    # output lat/lon
    out_proj = pyproj.Proj(init='epsg:4326')
    # transform xy to lat/lon
    lon, lat = pyproj.transform(in_proj, out_proj, x, y)
    return (lon, lat)


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


def get_satellite_subset(ds_all, r_start, r_end, c_start, c_end):
    '''
    Description:
        Returns a numpy data array of the rasted satellite
        image for all bands from a gdal object.
    Inputs:
        ds_all (osgeo.gdal.Dataset): Gdal data structure from opening a tif,
            ds_all = gldal.Open('...')
        r_start (int): Initial row pixel number of subset
        c_start (int): Initial column pixel number of subset
        r_end (int): Initial row pixel number of subset
        c_end (int): Final row pixel number of subset
    Returns:
        band_data (np.array, size=[r_del, c_del, 4]): The rastered image
            data for all bands
    Updates:
        N/A
    Write to file:
        N/A
    '''
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
    for index in np.arange(channels):
        band_num = index + 1
        data = ds_all.read(band_num,
                           window=((r_start, r_end), (c_start, c_end)))
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


def find_peaks_for_subset(ds_all, r_start, r_end, c_start, c_end, plot_flag):
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
    band_data = get_satellite_subset(ds_all, r_start, r_end, c_start, c_end)
    # get tree data
    plant_data = get_tree_finder_image(band_data)
    # get peaks
    trees_local = detect_peaks(plant_data)
    # plot it
    if plot_flag:
        plot_satellite_image(band_data, plant_data, tree_loc)
    # store output
    plant_dict = {}
    leading_zeros = int(np.ceil(np.log10(np.max([ds_all.width,
                                                 ds_all.height]))))
    # build up string
    id_base = 'sat_'
    int_str_format = '{:0' + str(leading_zeros) + '}'
    # store the end of the range
    store_id = (id_base +
                int_str_format.format(r_start) + '_' +
                int_str_format.format(r_end) + '_' +
                int_str_format.format(c_start) + '_' +
                int_str_format.format(c_end) + '_'
                )
    # put peaks in a nice format
    trees_global = np.zeros(np.shape(trees_local))
    trees_global[:, 0] = trees_local[:, 0] + r_start
    trees_global[:, 1] = trees_local[:, 1] + c_start
    # grab lat/lon
    trees_lonlat = np.zeros(np.shape(trees_local))
    (trees_lonlat[:, 0], trees_lonlat[:, 1]) = (
        proj_rc_2_latlon(trees_global[:, 0], trees_global[:, 1],
                         ds_all))
    # store it
    plant_dict = {'store_id': store_id, 'r_start': r_start, 'r_end': r_end,
                  'c_start': c_start, 'c_end': c_end,
                  'trees_local': trees_local,
                  'trees_global': trees_global,
                  'trees_lonlat': trees_lonlat}
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
    # set-up logger
    logger = logging.getLogger('sat')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('sat.log', mode='w')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    # start timer
    t = TicToc()
    t.tic()
    # log it
    logger.info('Reading in file: ' + args.sat_file)
    # get tif data
    ds_all = rasterio.open(sat_file)
    # get raster bands for a subset
    r_start = ds_all.height // 2
    c_start = ds_all.width // 2
    r_del = 1000
    c_del = 1000
    r_end = r_start + 2 * r_del
    c_end = c_start + 2 * c_del
    counter = 0
    tree_coords_all = []
    tree_lonlat_all = []
    # loop over subsets
    for rs in np.arange(r_start, r_end, r_del):
        for cs in np.arange(c_start, c_end, c_del):
            tree_dict = find_peaks_for_subset(ds_all, rs, rs+r_del,
                                              cs, cs + c_del, plot_flag)
            # store just the global tree coordinates
            if counter == 0:
                tree_coords_all = tree_dict['trees_global']
                tree_lonlat_all = tree_dict['trees_lonlat']
            else:
                tree_coords_all = np.append(tree_coords_all,
                                            tree_dict['trees_global'],
                                            axis=0)
                tree_lonlat_all = np.append(tree_lonlat_all,
                                            tree_dict['trees_lonlat'],
                                            axis=0)
            counter += 1
    # dump it
    pickle.dump((tree_coords_all, tree_lonlat_all),
                open('tree_coords.pkl', 'wb'))
    # store time
    run_time = t.tocvalue()
    logger.info('Run time ' + str(run_time) + ' sec')


if __name__ == '__main__':
    '''
    Description:
        Takes in commandline arguements and self main()
    Example call:
        python src/satellite_analyze data/raw/athens_satellite.tif --plot=true
    '''
    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('sat_file', type=str, help='path to satellite tif')
    parser.add_argument('--plot', type=bool, default=False, help='plot flag')
    args = parser.parse_args()
    # run main
    main(args.sat_file, args.plot)
