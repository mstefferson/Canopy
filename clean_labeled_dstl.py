import clean_dstl
import label_dstl
import argparse
import logging
import json
import os
import warnings


def main(config, logger):
    '''
    The yolo model does something strange to the data. This is an outstanding bug,
        but this function is a temporary work around.
    Args:
        config (dict): loaded dstl config json. Contains image and path info
    Returns:
        N/A
    Update:
        N/A
    Writes to file:
        Writes /path/2/processed/data/(train, val)/(images, labels)
    '''
    # make dirs
    if not os.path.exists(os.getcwd() + config["dstl"]["proc_data_rel"]):
        os.makedirs(os.getcwd() + config["dstl"]["proc_data_rel"])
    # set paths
    data_path = os.getcwd() + config["dstl"]["raw_data_rel"]
    save_path = os.getcwd() + config["dstl"]["proc_data_rel"]
    # ignore low contrast warning
        print('Training')
        path_images = os.getcwd() + '/data/processed/dstl/train/images'
        path_label = os.getcwd() + '/data/processed/dstl/train/labels'
        label_dstl.verify_image_label_match(path_images, path_label)
        print('Valid')
        path_images = os.getcwd() + '/data/processed/dstl/valid/images'
        path_label = os.getcwd() + '/data/processed/dstl/valid/labels'
        label_dstl.verify_image_label_match(path_images, path_label)


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
    # set-up logger
    logger = logging.getLogger('dstl')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('dstl_clean.log', mode='w')
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
    main(config, logger)
