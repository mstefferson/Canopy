import numpy as np


def divide_image_2_regions(num_r, num_c, num_div_r, num_div_c):
    # divide up bos into regions
    delta_r = np.floor(float(num_r) / float(num_div_r))
    delta_c = np.floor(float(num_c) / float(num_div_c))
    regions_r = np.repeat(np.arange(num_div_r), delta_r)[0:num_r]
    regions_c = np.repeat(np.arange(num_div_c), delta_c)[0:num_c]
    return regions_r, regions_c


def put_labels_2_regions(num_r, num_c, num_reg_r, num_reg_c, y):
    # get regions
    (reg_map_r, reg_map_c) = divide_image_2_regions(num_r, num_c,
                                                    num_reg_r, num_reg_c)
    # convert x, y to region
    reg_r = reg_map_r[(y[:, 1] * num_r).astype('int')]
    reg_c = reg_map_c[(y[:, 2] * num_c).astype('int')]
    reg_coors = [(x, y) for (x, y) in zip(reg_r, reg_c)]
    return reg_coors


def built_out_from_file(filename):
    num_r = 200
    num_c = 200
    num_reg_r = 25
    num_reg_c = 25
    label_mat = parse_labels(filename)
    y = build_outcome_vecs(label_mat, 16)
    reg_coors = put_labels_2_regions(num_r, num_c, num_reg_r,
                                     num_reg_c, y)
    output = reg_coors_2_output(reg_coors, num_reg_r, num_reg_c, y)
    return output


def reg_coors_2_output(reg_coors, num_r, num_c, y):
    '''
    Take outputs for each label, y, and put them as the outputs
    for each output bounding box region
    '''
    num_2_fill = len(reg_coors)
    output = np.zeros((num_r, num_c, np.shape(y)[1]))
    for index in np.arange(num_2_fill):
        output[reg_coors[index][0], reg_coors[index][1], :] = y[index, :]
    return output


def parse_labels(filename):
    '''
    Put labels into a matrix (num_label, 5)
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


def build_outcome_vecs(label_mat, num_labels):
    '''
    Take the parsed labels contained in label mat, and put them into
    a matrix of outcome vectors that will be assigned to bounding box
    regions
    '''
    # allocate for outcome vec
    outcome_vecs = np.zeros((np.shape(label_mat)[0], num_labels+5))
    # first index 1 (100% prob of object)
    outcome_vecs[:, 0] = 1
    # next elements are lo
    print(np.shape(outcome_vecs), np.shape(label_mat[:, 1:]))
    outcome_vecs[:, 1:5] = label_mat[:, 1:]
    # set class to one based on label
    classes_2_set = label_mat[:, 0].astype('int')
    outcome_vecs[:, classes_2_set] = 1
    return outcome_vecs


def cost_function(y, y_pred):
    '''
    Squared distance cost function
    '''
    # based cost on is box is true or not
    cost = np.zeros((np.shape(y[:, :, 0])))
    mask_obj_true = y[:, :, 0] == 1
    # just take l2
    l2 = (y - y_pred) ** 2
    cost_regions = l2[:, :, 0] + (1-y[:, :, 0]) * np.sum(l2, axis=2)
    cost = np.sum(cost_regions)
    return cost, cost_regions
