import sys
import argparse
import json
sys.path.insert(0, "src")
import sat_class


def main(config):
    '''
    Collects all the predicted outputs and saves them
        a csv

    Args:
        config (dict): config dictionary

    Returns:
        N/A

    Updates:
        N/A

    Writes to file:
        Write collected outputs to config["sat_info"]["output_file_name"]
    '''
    # get params
    # collect it
    sat_master = sat_class.SatelliteTif(
        tif_file=config["sat_info"]["tif_file"],
        rel_path_2_data=config["sat_info"]["processed_data_path"],
        rel_path_2_output=config["sat_info"]["output_path"],
        imag_w=config["sat_info"]['imag_w'],
        imag_h=config["sat_info"]["imag_h"])
    print('collecting outputs')
    sat_master.collect_outputs(config["sat_info"]["output_file_name"])


if __name__ == '__main__':
    '''
    Executeable:
    python src/models/keras_yolo2/predict.py \
    -c configs/config_satfile.json \
    '''
    # set-up arg parsing
    argparser = argparse.ArgumentParser(
        description='Collect files from predictions')
    argparser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
    args = argparser.parse_args()
    # set configs
    config_path = args.conf
    # build config
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    # run main
    main(config)
