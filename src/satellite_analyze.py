import logging
import os
import glob
import numpy as np
import argparse
from pytictoc import TicToc
import pickle
import geopandas
import pyproj
import rasterio
import src.models
import imageio


def get_lonlat_tree_file(filename):
    '''
    Given a GIS *shp file, return the lat/lon of the recorded points

    Args:
        filename (str): Path to *shp file

    Returns:
        lonlat (np.array shape=[n, 2]): lon/lat of all recorded points

    Updates:
        N/A

    Write to file:
        N/A
    '''
    # Read data using geopandas
    data = geopandas.read_file(filename)
    # map data to lat/long using projection type epsg:4326
    data = data.to_crs({"init": "epsg:4326"})
    # get lat from lon from bounds of point object
    lon = data['geometry'].apply(lambda x: x.bounds[0])
    lat = data['geometry'].apply(lambda x: x.bounds[1])
    # store it in numpy array
    lonlat = np.array([lon, lat]).transpose()
    return lonlat


def get_xy_tree_file(filename):
    '''
    Given a GIS *shp file, return the x/y (geospatial coords)
        of the recorded points

    Args:
        filename (str): Path to *shp file

    Returns:
        xy (np.array shape=[n, 2]): geospatial coords of all recorded points

    Updates:
        N/A

    Write to file:
        N/A
    '''
    # Read data using geopandas
    data = geopandas.read_file(filename)
    # get lat from lon from bounds of point object
    x = data['geometry'].apply(lambda x: x.bounds[0])
    y = data['geometry'].apply(lambda x: x.bounds[1])
    # store it in numpy array
    xy = np.array([x, y]).transpose()
    return xy


def get_known_tree_all(transfunc=get_lonlat_tree_file):
    '''
    Given a GIS *shp file, return the x/y (geospatial coords)
        of the recorded points

    Args:
        filename (str): Path to *shp file

    Returns:
        xy (np.array shape=[n, 2]): geospatial coords of all recorded points

    Updates:
        N/A

    Write to file:
        N/A
    '''
    root = '/app/data/raw/sv_gis/sv_gis_2017/'
    file1 = root + 'dendra.shp'
    file2 = root + 'glastra.shp'
    file3 = root + 'pagkakia.shp'
    # get all the data
    data1 = transfunc(file1)
    data2 = transfunc(file2)
    data3 = transfunc(file3)
    # collect all data together
    data_all = np.append(data1, np.append(data2, data3, axis=0), axis=0)
    return data_all


def proj_lonlat_2_rc(lon, lat, dataset):
    '''
    Convert lon/lat coordinates to rows and columns in the tif satellite
        image. Uses pyproj to convert between coordinate systems

    Args:
        lon (float): longitude
        lat (float): latitude
        dataset (rasterio.io.DatasetReader): Gdal data structure from opening a
            tif, dataset = rasterio.open('...')

    Returns:
        rc (np.array shape=[n, 2]): row/columns in tif file for all
            recorded points

    Updates:
        N/A

    Write to file:
        N/A
    '''
    # input lat/lon
    in_proj = pyproj.Proj(init='epsg:4326')
    # output based on crs of tif/shp
    out_proj = pyproj.Proj(dataset.crs)
    # transform lat/lon to xy
    x, y = pyproj.transform(in_proj, out_proj, lon, lat)
    # convert rows and columns to xy map project
    (r, c) = rasterio.transform.rowcol(dataset.transform, x, y)
    # store it in numpy array
    rc = np.array([r, c]).transpose()
    return rc


def proj_rc_2_lonlat(r, c, dataset):
    '''
    Convert row/columns of tif dataset to lat/lon.
        Uses pyproj to convert between coordinate systems

    Args:
        lon (float): longitude
        lat (float): latitude
        dataset (rasterio.io.DatasetReader): Gdal data structure from opening a
            tif, dataset = rasterio.open('...')

    Returns:
        rc (np.array shape=[n, 2]): row/columns in tif file for all
            recorded points

    Updates:
        N/A

    Write to file:
        N/A
    '''
    # convert rows and columns to xy map project
    (x, y) = rasterio.transform.xy(dataset.transform, r, c)
    # input based on crs of tif/shp
    in_proj = pyproj.Proj(dataset.crs)
    # output lat/lon
    out_proj = pyproj.Proj(init='epsg:4326')
    # transform xy to lat/lon
    lon, lat = pyproj.transform(in_proj, out_proj, x, y)
    # store it in numpy array
    lonlat = np.array([lon, lat]).transpose()
    return lonlat


def build_test_train(sat_file, num_images, delta=200,
                     split=0.3, c_channels=[0, 1, 3], fid_start=0):
    '''
    Builds a test/train set from random squares of a sat_file tif file

    Args:
        sat_file (str): path to satellite tif file
        num_images (int): total number of images in data set
        delta (int, optional): width/heigth of image in data set
        split (float, optional): test/train split (frac in test)
        c_channels (list of ints, optional): three color bands
            to include in image. Tiff has four: r, g, b, IR
        fid_start: initial file id to label images. e.g., image_fid.jpg

    Returns:
        N/A

    Updates:
        N/A

    Write to file:
        /app/data/(test,train)/images/image_*,
        /app/data/(test,train)/images/key*: test and train image data
            set with keys
    '''
    def save_data(sat_data, coors, num2save, num_total, path, save_str, fid):
        '''
        Description:
            Nested function that actually writes the files
        '''
        look_up_name = 'key'
        look_up_f = open(path + look_up_name + '.txt', 'w+')
        look_up_dict = {}
        for ii in np.arange(num2save):
            # get subset
            band_data = get_satellite_subset(sat_data,
                                             coors[ii, 0],
                                             coors[ii, 1],
                                             coors[ii, 2],
                                             coors[ii, 3])
            # update band to save based on color channels
            # must be saved as integars from 0, 255
            band2save = np.array(255*band_data[:, :, c_channels],
                                 dtype=np.uint8)
            # save everything
            img_save_name = save_str.format(fid+ii) + '.jpg'
            # same image
            imageio.imwrite(path + img_save_name, band2save)
            # update lookup dict
            look_up_dict[img_save_name] = train_coors[ii, :]
            # update lookup txt file
            line2save = (img_save_name
                         + ' ' + str(coors[ii, 0])
                         + ' ' + str(coors[ii, 1])
                         + ' ' + str(coors[ii, 2])
                         + ' ' + str(coors[ii, 3])
                         + '\n')
            look_up_f.write(line2save)
        # save lookup dictionary
        pickle.dump(look_up_dict, open(path + look_up_name + '.pkl', 'wb'))
        # close txt file
        look_up_f.close()
    # build output directories if they don't exist
    directory = '/app/data/train/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = '/app/data/test/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # get sat data
    sat_data = rasterio.open(sat_file)
    # get number of rows and columns
    num_r = sat_data.height
    num_c = sat_data.width
    # find all possible subsections
    r_min = num_r / 4
    r_max = 3*num_r / 4
    c_min = num_c / 4
    c_max = 3*num_c / 4
    # get all possible row/column starting points
    r_start_all = np.arange(r_min, r_max, delta)
    c_start_all = np.arange(r_min, r_max, delta)
    # get random sample
    r_start = np.random.choice(r_start_all, num_images)
    c_start = np.random.choice(c_start_all, num_images)
    # coordinates
    start_coord = np.array(
        [r_start, c_start]).transpose()[:num_images].astype('int')
    # build full array (r_start, r_end, c_start, c_end)
    all_coors = np.zeros((num_images, 4))
    all_coors[:, 0] = start_coord[:, 0]
    all_coors[:, 1] = start_coord[:, 0] + delta
    all_coors[:, 2] = start_coord[:, 1]
    all_coors[:, 3] = start_coord[:, 1] + delta
    # split into test/train
    all_coors = all_coors.astype('int')
    num_test = int(np.floor(num_images * split))
    num_train = int(num_images - num_test)
    train_coors = all_coors[:num_train, :]
    test_coors = all_coors[num_train:, :]
    # set str format for save name
    num_dec = int(np.ceil(np.log10(fid + num_total))) + 1
    save_str = 'image_{:0' + str(num_dec) + 'd}'
    # save train
    train_path = '/app/data/train/images/'
    save_data(sat_data, train_coors, num_train,
              num_images, train_path, save_str, fid)
    # save test
    test_path = '/app/data/test/images/'
    save_data(sat_data, test_coors, num_test,
              num_images, test_path, save_str, fid+num_train)


def get_satellite_subset(ds_all, r_start, r_end, c_start, c_end,
                         norm=None):
    '''
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
    # All color bands peak at white light (which building are), get a average
    # to sub it out
    # handle a three vs four bands differently
    # 3 bands: r, g, IR
    if np.shape(band_data)[2] == 3:
        data_ave = ((band_data[:, :, 0] + band_data[:, :, 1])
                    / 2)
        # Plants reflect green and IR so take a weighted average and subtract
        # out background. This weighted average is just based on messing
        # around with the data
        plant_data = (3*band_data[:, :, 1] + band_data[:, :, 2]) / 4 - data_ave
    # 4 bands: r, g, b, IR
    else:
        data_ave = ((band_data[:, :, 0] + band_data[:, :, 1] +
                     band_data[:, :, 2]) / 3)
        # Plants reflect green and IR so take a weighted average and subtract
        # out background. This weighted average is just based on messing
        # around with the data
        plant_data = (3*band_data[:, :, 1] + band_data[:, :, 3]) / 4 - data_ave
    plant_data[plant_data < drop_thres] = 0
    return plant_data


def find_peaks_for_subset(ds_all, r_start, r_end, c_start, c_end):
    '''
    Grabs a data subset and finds all the trees. This is a wrapper
        for many functions that: grab subset, gets plant data, find trees

    Args:
        ds_all (osgeo.gdal.Dataset): Gdal data structure from opening a tif,
            ds_all = gldal.Open('...')
        r_start (int): Initial row pixel number of subset
        c_start (int): Initial column pixel number of subset
        r_del (int): Width of subset in pixels along rows
        c_del (int): Width of subset in pixels along columns

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
    trees_local = src.models.detect_peaks(plant_data)
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
        proj_rc_2_lonlat(trees_global[:, 0], trees_global[:, 1],
                         ds_all))
    # store it
    plant_dict = {'store_id': store_id, 'r_start': r_start, 'r_end': r_end,
                  'c_start': c_start, 'c_end': c_end,
                  'trees_local': trees_local,
                  'trees_global': trees_global,
                  'trees_lonlat': trees_lonlat}
    return plant_dict


def main(sat_file):
    '''
    Loops over an entire satellite image, divides it into subset,
        and finds trees for each subset. Writes a list of all
        tree locations (in pixels) to files as a pkl

    Args:
        sat_file (str): Path to satellite tif file

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
    r_del = 350
    c_del = 350
    r_end = r_start + r_del
    c_end = c_start + c_del
    counter = 0
    tree_coords_all = []
    tree_lonlat_all = []
    # loop over subsets
    for rs in np.arange(r_start, r_end, r_del):
        for cs in np.arange(c_start, c_end, c_del):
            tree_dict = find_peaks_for_subset(ds_all, rs, rs+r_del,
                                              cs, cs + c_del)
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
    Takes in commandline arguements and self main()

    Example call:
        python src/satellite_analyze data/raw/athens_satellite.tif
    '''
    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('sat_file', type=str, help='path to satellite tif')
    args = parser.parse_args()
    # run main
    main(args.sat_file)
