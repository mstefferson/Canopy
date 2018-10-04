from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import src.models.analyze_model


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


def predict_bounding_box(image):
    # predict to get boxes
    peaks = detect_peaks(image)
    df_cols = ['label', 'imag_w', 'imag_h',
               'x', 'y', 'w', 'h', 'conf']
    box_df = pd.DataFrame(index=np.arange(len(bboxes)),
                          columns=df_cols)
    for (row, box) in enumerate(bboxes):
        x_center = peaks[row, 1] / image.shape[1]
        width = 0
        y_center = peaks[row, 0] / image.shape[0]
        height = 0
        label = 0
        conf = 1
        box_df.loc[row, df_cols] = [label, image.shape[1], image.shape[0],
                                    x_center, y_center, width, height, conf]
    return box_df


def draw_boxes(image, box_df):
    # turn pixels of tree red
    for _, row in box_df.iterrows():
        row_tree = box_df['y'] * box_df['imag_h']
        col_tree = box_df['x'] * box_df['imag_w']
        min_r_tree = np.min([0, row_tree - 3])
        max_r_tree = np.max([box_df['imag_h'], row_tree + 3])
        min_c_tree = np.min([0, col_tree - 3])
        max_c_tree = np.max([box_df['imag_w'], col_tree + 3])
        image[min_r_tree:max_r_tree, :, 0] = 255
        image[:, min_c_tree:max_c_tree, 0] = 255
        return image


def main(config):
    '''
    Predict the bounding boxes for a single image
    Args:
        args (argparser.parse_args object): argument object with attibutes:
            args.conf: config file
            args.weights: weight to trained yolo2 model
            args.input: image file to predict on
            args.bound: boolean to write bounding boxes to file
            args.detect: boolean to draw bounding boxes image
    Returns:
        N/A
    Updates:
        N/A
    Writes to file:
        If flags are set, writes detected image (image with bounding boxes)
            and the bounding box locations to file. The outputs are located
            in directories with the same base path as the images,
            /base/path/images
    '''
    # get params
    weights_path = config["predict"]["weights"]
    save_detect = config["predict"]["save_detect"]
    save_bb = config["predict"]["save_bb"]
    # build model
    yolo_model = build_model(config, weights_path)
    # get all the files you want to predict on
    pred_path = config["predict"]["image_path"]
    # pred path can be a folder or image. Grab files accordingly
    if os.path.isfile(pred_path):
        files_2_pred = [pred_path]
    else:
        files_2_pred = glob.glob(pred_path + '*')
    print('Predicting objects on files', files_2_pred)
    # loop over all files
    for image_path in files_2_pred:
        # load image and predict bounding box
        image = cv2.imread(image_path)
        # build bounding boxes
        box_df = predict_bounding_box(image)
        if save_bb:
            # build file names and directories
            result_dir = config["predict"]["bb_folder"]
            path2write = base_dir + '/' + result_dir
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            filename = path2write + file_id + '.csv'
            box_df.to_csv(filename, index=False)
        if save_detect:
            # build file names and directories
            result_dir = config["predict"]["detect_folder"]
            path2write = base_dir + '/' + result_dir
            filename = (path2write + file_id +
                        '_detected' + file_name[-4:])
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            image = draw_boxes(image, box_df)
            cv2.imwrite(filename, image)
