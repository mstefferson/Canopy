import sys
import argparse
import json
import logging
import numpy as np
sys.path.insert(0, "src")
import sat_class


def main(config):
    '''
    Handles the cleaning and labeling of dstl images (equivalent to running
        clean_dstl and label_dstl. This will build a nice labeled training
        and validation set with the appropriate format. This wraps
        clean_dstl and label_dstl
    Args:
        config (dict): loaded dstl config json. Contains image and path info
    Returns:
        N/A
    Update:
        N/A
    Writes to file:
        Writes /path/2/processed/data/(train, val)/(images, labels)
    '''
    # set-up logger
    logger = logging.getLogger('sat_build')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('sat_build.log', mode='w')
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
    # build the sat_obj
    sat_master = sat_class.SatelliteTif(
        tif_file=config["sat_info"]["tif_file"],
        rel_path_2_data=config["sat_info"]["processed_data_path"],
        rel_path_2_output=config["sat_info"]["output_path"],
        c_channels=config["sat_info"]["c_channels"],
        imag_w=config["sat_info"]["imag_w"],
        imag_h=config["sat_info"]["imag_h"],
        train_window=config["sat_info"]["train_window"],
        num_train=config["sat_info"]["num_train"],
        valid_frac=config["sat_info"]["valid_frac"],
        r_pred_start=config["sat_info"]["r_pred_start"],
        r_pred_end=config["sat_info"]["r_pred_end"],
        c_pred_start=config["sat_info"]["c_pred_start"],
        c_pred_end=config["sat_info"]["c_pred_end"]
    )
    logger.info('Built satellite class')
    # let the user know what's going to get built
    all_dirs = np.array(['train', 'test', 'predict'])
    log_ind = [config["build_info"]["train"],
               config["build_info"]["valid"],
               config["build_info"]["predict"]]
    dir2build = all_dirs[log_ind]
    if config["build_info"]["warnings"]:
        print('Warning: going to delete and rebuild ' + str(dir2build) +
              ' hit enter to continue')
        input()
    if config["build_info"]["train"]:
        sat_master.clean_train_images()
        sat_master.build_train_dataset()
        logger.info('Built training dataset')
    if config["build_info"]["valid"]:
        sat_master.clean_valid_images()
        sat_master.build_valid_dataset()
        logger.info('Built validation dataset')
    if config["build_info"]["predict"]:
        sat_master.clean_pred_images()
        sat_master.build_pred_dataset()
        logger.info('Built prediction dataset')


if __name__ == '__main__':
    '''
    Executeable:
    python src/preprocess/clean_dstl/build_dstl_dataset.py /
        -c configs/config_dstl.json
    '''
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        help='path to config file')
    args = parser.parse_args()
    config_path = args.config
    # load config
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    main(config)
