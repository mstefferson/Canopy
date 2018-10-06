def draw_boxes(image, box_df):
    """
    Given an image, draw red pixels where the trees are located.
        This assumes the image was read in using cv2

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


def detect_peaks(array_with_peaks):
    """
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
    # 4 bands: r, g, b, IR
    data_ave = ((band_data[:, :, 0] + band_data[:, :, 1] +
                 band_data[:, :, 2]) / 3)
    # Plants reflect green and IR so take a weighted average and subtract
    # out background. This weighted average is just based on messing
    # around with the data
    plant_data = (3*band_data[:, :, 1] + band_data[:, :, 3]) / 4 - data_ave
    plant_data[plant_data < drop_thres] = 0
    return plant_data
