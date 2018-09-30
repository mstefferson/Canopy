import clean_dstl
import label_dstl
import argparse
import json
import os
import warnings


def main(config):
    '''
    Handles the cleaning and labeling of dstl images (equivalent to running
        clean_dstl and label_dstl. This will build a nice labeled training    
        and validation set with the appropriate format. This wraps
        clean_dstl and label_dstl
    Args:
        config (dist): loaded dstl config json. Contains image and path info
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clean_dstl.process_dstl_directory(
            dir_path=data_path + 'three_band/',
            sub_dirs=config["dstl"]["sub_dirs"],
            image_save_dir=save_path + 'chopped_images',
            annotations_save_dir=save_path,
            geojson_dir=geojson_dir,
            grid_sizes=grid_sizes,
            block_shape=(config["dstl"]["imag_h"],
                         config["dstl"]["imag_w"], 3)
        )
    print('Processed all data')
    # get all the bound box labels
    df, files2delete = label_dstl.get_all_bounding(
        config['dstl']['proc_data_rel'], config['dstl']['imag_w'],
        config['dstl']['imag_h'])
    print('Got all bounding boxes')
    # get all the bound box labels
    label_dstl.build_labels(df, files2delete, config['dstl']['imag_w'],
                            config['dstl']['imag_h'],
                            lab_format=config['dstl']['label_format'])
    print('Built all labels')
    # build val/train
    path2data = os.getcwd() + config['dstl']['proc_data_rel']
    label_dstl.build_val_train(path2data, val_size=0.3)
    print('Move to train/val')


if __name__ == '__main__':
    '''
    Executeable:
    python3 src/preprocess/clean_dstl/build_dstl_dataset.py /
        configs/config_dstl.json
    '''

    # parse inputs
    parser = argparse.ArgumentParser()
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='config file')
    args = parser.parse_args()
    config_path = args.config
    # load config
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    main(config)
