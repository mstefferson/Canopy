import clean_dstl
import label_dstl
import argparse
import logging
import json
import os
import warnings


def main(config, logger):
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
    # make dirs
    if not os.path.exists(os.getcwd() + config["dstl"]["proc_data_rel"]):
        os.makedirs(os.getcwd() + config["dstl"]["proc_data_rel"])
    # set paths
    data_path = os.getcwd() + config["dstl"]["raw_data_rel"]
    save_path = os.getcwd() + config["dstl"]["proc_data_rel"]
    geojson_dir = data_path + "train_geojson_v3/"
    grid_file = data_path + "grid_sizes.csv"
    grid_sizes = clean_dstl.import_grid_sizes(grid_file)
    # process the image
    # ignore low contrast warning
    # Loop over sub_dirs in case something breaks
    for a_dir in config["dstl"]["sub_dirs"]:
        logger.info('Analyzing dir ' + a_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clean_dstl.process_dstl_directory(
                dir_path=data_path + 'three_band/',
                sub_dirs=[a_dir],
                image_save_dir=save_path + 'chopped_images',
                annotations_save_dir=save_path,
                geojson_dir=geojson_dir,
                grid_sizes=grid_sizes,
                block_shape=(config["dstl"]["imag_h"],
                             config["dstl"]["imag_w"], 3)
            )
    logger.info('Processed all data')
    logger.info('Starting logging')
    # run label_dstl main
    label_dstl.main(config, logger)


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
    fh = logging.FileHandler('dstl_build.log', mode='w')
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
