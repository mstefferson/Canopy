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
