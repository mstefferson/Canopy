from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import src.analyze_model


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


def pixel_detect_model(array_with_peaks):
    """
    Description:
        Object locatization model built around peak detection
    Inputs:
        array_with_peaks (np.array): 2d array to find peaks of
    Returns:
        bb_output (np.array, size=[num_r_out, num_c_out, n]): output array
            for each bounding box regional. n is the length of the output
            array, n = (5+num_classes). Note, no anchor boxes!!!
    Updates:
        N/A
    Write to file:
        N/A
    """
    # get size of array
    (num_r, num_c) = np.shape(array_with_peaks)
    # set number of regions to something arbitrary
    num_reg_r = 25
    num_reg_c = 25
    # get peaks from  peak detect
    peaks = detect_peaks(array_with_peaks)
    num_peaks = np.shape(peaks)[0]
    # get position and have fake widths/heights
    x = peaks[:, 0] / num_r
    y = peaks[:, 1] / num_c
    # make up random widths/heights
    w = 0.03 * np.ones_like(peaks[:, 0])
    h = 0.03 * np.ones_like(peaks[:, 0])
    # build a fake data set just like
    # a labeled file
    num_classes = len(open('/app/data/train/labels/classes.txt').readlines())
    tree_guess = np.zeros((num_peaks, 5))
    # Build a matrix liek you'd get from labels
    tree_guess[:, 0] = num_classes
    tree_guess[:, 1] = x
    tree_guess[:, 2] = y
    tree_guess[:, 3] = w
    tree_guess[:, 4] = h
    print('Predicting {} trees'.format(num_peaks))
    y = src.analyze_model.build_outcome_vecs(tree_guess, 16)
    # get regions
    (reg_r, reg_c) = src.analyze_model.build_in_out_region_map(num_r, num_c,
                                                               num_reg_r,
                                                               num_reg_c)
    # convert x, y to region
    reg_r = reg_map_r[(tree_guess[:, 1] * num_r).astype('int')]
    reg_c = reg_map_c[(tree_guess[:, 2] * num_c).astype('int')]
    # build output vector
    bb_output = output_vec_2_to_bb_output(y, reg_r, reg_c,
                                          num_reg_r, num_reg_c)
    return bb_output
