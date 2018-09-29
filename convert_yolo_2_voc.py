import os
import sys
import shutil
import glob
import argparse
import numpy as np
from PIL import Image
from lxml import etree


def convert_all_yolo(yolo_lab_path, voc_path, imag_w, imag_h):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"): 
            full_path = os.path.join(directory, filename)
            print(full_path)
            annotation = write_voc_file('blah',
                                        ['trees', 'poop'],
                                        [[0.1, 0.11, 0.2, 0.21],
                                         [0.3, 0.31, 0.4, 0.41]],
                                        imag_w,
                                        imag_h)
        else:
            print("Do not recognize file type")


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='config file')
    args = parser.parse_args()
    config_path = args.config
    print(os.getcwd())
    # load config
    with open(config_path) as config_buffer:
        # config = json.loads(config_buffer.read())
        config = json.load(config_buffer)

