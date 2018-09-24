import numpy as np
import rasterio
import src.analyze_model
import src.models
import src.satellite_analyze


def divide_tiff(sat_w, sat_h, image_w=200, image_h=200):
    # imagesize = [height, width]
    # get number of divisions
    div_w = int(np.floor(sat_w / image_w))
    div_h = int(np.floor(sat_h / image_h))
    # get trim off edges
    trim_w = sat_w - div_w*image_w
    trim_h = sat_h - div_h*image_h
    # get start indices (columns/width, rows/height)
    start_c = np.floor(trim_w/2)
    end_c = sat_w - np.ceil(trim_w/2)
    start_r = np.floor(trim_h/2)
    end_r = sat_h - np.ceil(trim_h/2)
    # get all origins
    row_origins = np.arange(start_r, end_r, image_w)
    col_origins = np.arange(start_c, end_c, image_h)
    return row_origins, col_origins


def pred_subset(sat_data, r_start, r_end, c_start, c_end,
                model='none', c_channels=[0, 1, 3]):
    # check to make sure it's not zero
    delta_r = r_end - r_start
    delta_c = c_end - c_start
    zero_compare = np.zeros((delta_r, delta_c, 3))
    # get band data
    band_data = src.satellite_analyze.get_satellite_subset(
        sat_data, r_start, r_end, c_start, c_end)
    band_data = band_data[:, :, c_channels]
    if np.all(band_data != zero_compare):
        # try and build an output for the tree data
        if model == 'pixel_detect':
            out_pred = src.models.pixel_detect_model(data)
        else:
            out_pred = None
    else:
        out_pred = None
    return out_pred


def pred_subset(sat_data, r_start, r_end, c_start, c_end,
                model='none', c_channels=[0, 1, 3]):
    # check to make sure it's not zero
    delta_r = r_end - r_start
    delta_c = c_end - c_start
    zero_compare = np.zeros((delta_r, delta_c, 3))
    # get band data
    band_data = src.satellite_analyze.get_satellite_subset(
        sat_data, r_start, r_end, c_start, c_end)
    band_data = band_data[:, :, c_channels]
    if np.all(band_data != zero_compare):
        predictions = np.random.rand(np.random.randint(3)+1, 5)
    else:
        predictions = None
    return predictions


def pred_tiff(sat_file,  r_start=0,
              r_end=np.inf, c_start=0, c_end=np.inf,
              image_w=200, image_h=200):
    # get data
    sat_data = rasterio.open(sat_file)
    sat_width = sat_data.width
    sat_height = sat_data.height
    scale_image = [image_h, image_w]
    # get orgins for each subset
    r_end = np.min([r_end, sat_data.height])
    c_end = np.min([c_end, sat_data.width])
    # get orgins
    origins_r, origins_c = divide_tiff(sat_width, sat_height,
                                       image_w=200, image_h=200)
    origins_r = origins_r[(origins_r > r_start) &
                          (origins_r < r_end)].astype(int)
    origins_c = origins_c[(origins_c > c_start) &
                          (origins_c < c_end)].astype(int)
    origin_list = [(r, c) for r in origins_r for c in origins_c]
    print('sat file ({}x{})'.format(sat_height, sat_width))
    print('Number of origins', len(origin_list))
    counter = 0
    # keep a list of all located trees
    tree_dict_info = []
    tree_info = np.empty((0, 4), int)
    for origin in origin_list:
        pred = pred_subset(sat_data,
                           origin[0], origin[0]+image_h,
                           origin[1], origin[1]+image_w)
        # make a list of all located trees
        pred_dict = {}
        pred_dict['origin'] = origin
        pred_dict['size'] = scale_image
        pred_dict['local'] = pred
        # put local prediction into global image
        if pred is not None:
            # convert to global
            pred_global = np.ones_like(pred)
            pred_global[:, 0] = (pred[:, 0] * scale_image[0]) + origin[0]
            pred_global[:, 1] = (pred[:, 1] * scale_image[1]) + origin[1]
            pred_global[:, 2] = (pred[:, 2] * scale_image[0])
            pred_global[:, 3] = (pred[:, 3] * scale_image[1])
            pred_global[:, 4] = pred[:, 4]
            # convert to int
            pred_dict['global'] = pred_global
            tree_info = np.append(tree_info, pred_global[:, :4].astype('int'),
                                  axis=0)
        else:
            pred_dict['global'] = None
        tree_dict_info.append(pred_dict)
        counter += 1
        if np.mod(counter, np.floor(len(origin_list)/20)) == 0:
            print('Percent done:', counter / len(origin_list))
    return tree_dict_info, tree_info
