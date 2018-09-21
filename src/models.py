import numpy as np


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
    w = 0.03 * np.oneslike(peaks[:, 0])
    h = 0.03 * np.oneslike(peaks[:, 0])
    # build a fake data set just like
    # a labeled file
    tree_guess = np.zeros(num_peaks, 5+1)
    # classification number 15 is tree currently
    tree_guess[:, 0] = 15
    tree_guess[:, 1] = x
    tree_guess[:, 2] = y
    tree_guess[:, 3] = w
    tree_guess[:, 4] = h
    outcome_vec = src.analayze_model(tree_guess, 16)
    # get regions
    (reg_r, reg_c) = divide_image_2_regions(num_r, num_c, num_reg_r, num_reg_c)
    # build bounding box output
    bb_out = put_labeled_2_bb_output(num_r, num_reg_c, num_reg_r,
                                     num_div_c, outcome_vec)
    return bb_out
