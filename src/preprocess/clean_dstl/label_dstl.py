import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='config file')
    args = parser.parse_args()
    config_path = parser.config
    # load config
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    df = pd.read_csv('annotations.csv', header=None)
    df.columns = ['file', 'x_min', 'y_min', 'x_max', 'y_max', 'label_str']
    ann_file = (os.getcwd() + config['dstl']['proc_data_rel'] +
                "annotations/annotations.csv")
    label_path = os.getcwd() + config['dstl']['proc_data_rel'] + "/yolo_labels"
    # make dir
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    df = pd.read_csv(ann_file, header=None)
    df.columns = ['file', 'x_min', 'y_min', 'x_max', 'y_max', 'label_str']
    # find and remove all nans
    files2delete = df.file.loc[df.isnull().any(axis=1)]
    print('Deleting {} files'.format(len(np.unique(files2delete))))
    # drop nans()
    df.dropna(axis=0, inplace=True)
    # grab files
    all_files = pd.unique(df['file'])
    # set all the labels, after rerunning set just to trees
    label_dir = {'trees': 0, 'buildings': 1, 'animals': 2}
    df['labels'] = df.label_str.apply(lambda x: label_dir[x]).astype('int')
    df['labels'] = df['labels'].astype('int')
    # get width and height
    w = df['x_max'] - df['x_min']
    h = df['y_max'] - df['y_min']
    width_image = 300
    height_image = 300
    x_center = (df['x_min'] + w / 2) / width_image
    y_center = (df['y_min'] + h / 2) / height_image
    df['w'] = w / width_image
    df['h'] = h / height_image
    df['x'] = x_center
    df['y'] = y_center
    for f_name in all_files:
        data_temp = df.loc[df['file'] == file]
        f = open(f_name, 'w+')
        for index, row in data_temp.iterrows():
            str2dump = str([row['label'], row['x'],
                            row['y'], row['w'], row['h']])
            f.write(str2dump + '\n')
        f.close()
        print(f_name)
        print(data_2_dump)
    break
