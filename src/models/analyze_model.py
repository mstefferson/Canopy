import numpy as np


def build_in_out_region_map(num_r_in, num_c_in, num_r_out, num_c_out):
    '''
    Description:
        Given an input image (num_r_in, num_c_in) and an bounding box
        output (num_r_out, num_c_out), build a map that can convert
        inputs row/cols to bound box rows/cols
    Args:
        num_r_in (int): number of rows in input
        num_c_in (int): number of columns in input
        num_r_out (int): number of rows in bounding box output
        num_c_out (int): number of columns in bounding box output
    Returns:
        region_map_r (np.array shape=[num_r_in,]): maps initial input rows
            to the binned rows of the output r_out = region_map_r[r_in]
        region_map_c (np.array shape=[num_c_in,]): maps initial input columns
            to the binned rows of the output c_out = region_map_c[c_in]
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # divide up input into regions
    delta_r = np.floor(float(num_r_in) / float(num_r_out))
    delta_c = np.floor(float(num_c_in) / float(num_c_out))
    # build a map from input rows to output rows
    region_map_r = np.repeat(np.arange(num_r_out), delta_r)[0:num_r_in]
    region_map_c = np.repeat(np.arange(num_c_out), delta_c)[0:num_c_in]
    return region_map_r, region_map_c


def output_vec_2_to_bb_output(y, reg_coors, num_r, num_c):
    '''
    Description:
        Given an output vector for each label, put it into
        the output matrix. Each label should go into the
        correct region.
    Args:
        y (np.array, size=[num_labels, n]): all of the labled output
            vectors, where n depends on the number of classes/anchor boxes
        num_r (int): number of rows of output bounding box matrix
        num_c (int): number of cols of output bounding box matrix
        reg_coors (np.array(num_trees, 2): the coordinates of the localized
            objects
    Returns:
        output (np.array, size=[num_r_out, num_c_out, n]): output array
            for each bounding box regional. n is the length of the output
            array, n = (5+num_classes). Note, no anchor boxes!!!
    Updates:
        N/A
    Write to file:
        N/A
    '''
    num_2_fill = len(reg_coors)
    output = np.zeros((num_r, num_c, np.shape(y)[1]))
    for index in np.arange(num_2_fill):
        output[reg_coors[index, 0], reg_coors[index, 1], :] = y[index, :]
    return output


def parse_labels(filename):
    '''
    Description:
        Parse the lableImg files (assumes YOLO format)
    Args:
        filename (str): path to label
    Returns:
        label_mat (np.array, size=[num_labels,  5]): output matrix
            for each label. Each row is separate label
            with values ('class', x, y, w, h)
    Updates:
        N/A
    Write to file:
        N/A
    '''
    labels = []
    # open file and grab lines
    labfile = open(filename, "r")
    for line in labfile.readlines():
        data = line.split()
        labels.append(data)
    # convert to a more useful format
    num_labels = len(labels)
    num_columns = len(labels[0])
    label_mat = np.zeros((num_labels, num_columns))
    for il in np.arange(num_labels):
        label_mat[il, :] = np.array(labels[il])
    return label_mat


def build_outcome_vecs(label_mat, num_classes):
    '''
    Description:
        Take the parsed labels contained in label mat, and put them into
        a matrix of outcome vectors that will be assigned to bounding box
        regions
    Args:
        label_mat (np.array, size=[num_labels,  5]): output matrix
            for each label
        num_classes (int): number of classes
    Returns:
        outcome_vec (np.array, size=[num_labels, n]): outcome
            vecs for each label, where n=num_classes+5
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # allocate for outcome vec
    outcome_vecs = np.zeros((np.shape(label_mat)[0], num_classes+5))
    # first index 1 (100% prob of object)
    outcome_vecs[:, 0] = 1
    # next elements are lo
    # set positions
    outcome_vecs[:, 1:4] = label_mat[:, 1:4]
    # set class to one based on label
    classes_2_set = label_mat[:, 0].astype('int')
    outcome_vecs[:, classes_2_set] = 1
    return outcome_vecs


def built_out_from_file(filename, num_r_in=200, num_c_in=200,
                        num_r_out=25, num_c_out=25):
    '''
    Description:
        Given a labeled file, build an output martix of the labeled
        bounding box regions.
    Args:
        filename (str): labeled file name
            (e.g., /app/data/test/label/image_*.txt)
        num_r_in (int, optional): number of rows in input must match image size
        num_c_in (int, optional): number of cols in input must match image size
        num_r_out (int, optional): number of rows in bounding box output
        num_c_out (int, optional): number of columns in bounding box output
    Returns:
        bb_output (np.array, size=[num_r_out, num_c_out, n]): output array
            for each bounding box regional. n is the length of the output
            array, n = (5+num_classes). Note, no anchor boxes!!!
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # parse the labels from file
    label_mat = parse_labels(filename)
    # convert the labels into output vectors
    num_classes = len(open('/app/data/train/labels/classes.txt').readlines())
    y = build_outcome_vecs(label_mat, num_classes)
    # Put the coordinates into the output regions
    # get map
    (reg_map_r, reg_map_c) = build_in_out_region_map(num_r_in, num_c_in,
                                                     num_r_out, num_c_out)
    # convert x, y to region
    reg_r = reg_map_r[(y[:, 1] * num_r_in).astype('int')]
    reg_c = reg_map_c[(y[:, 2] * num_c_in).astype('int')]
    reg_coors = np.array([reg_r, reg_c]).transpose()
    # build output vector
    bb_output = output_vec_2_to_bb_output(y, reg_coors,
                                          num_r_out, num_c_out)
    return bb_output


def cost_function(y_bb_true, y_bb_pred):
    '''
    Description:
        Squared distance cost function. Handles error for each
        region differently depending if an object is there or not
    Args:
        y_bb_true (np.array, size=[m, n, c]): The true bounding box
            object localization where m,n are row/cols on bounding
            box matrix and c is the output vector
        y_bb_pred (np.array, size=[m, n, c]): The prediction for bounding box
            object localization where m,n are row/cols on bounding
            box matrix and c is the output vector
    Returns:
        cost (float): total cost
        cost_regions (np.array, size=[m, n]): Cost for each region
    Updates:
        N/A
    Write to file:
        N/A
    '''
    # based cost on is box is true or not
    cost = np.zeros((np.shape(y_bb_true[:, :, 0])))
    mask_obj_true = y_bb_true[:, :, 0] == 1
    # all error currently l2 difference
    l2 = (y_bb_true - y_bb_pred) ** 2
    # determine the cost per region differently depending
    # if an object is there or not
    cost_regions = l2[:, :, 0] + (1-y_bb_true[:, :, 0]) * np.sum(l2, axis=2)
    # total cost
    cost = np.sum(cost_regions)
    return cost, cost_regions
