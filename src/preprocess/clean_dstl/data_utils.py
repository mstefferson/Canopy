"""Base functions used to handle raw data being loaded, cleaned, and resaved

Code written by Ben Hammel (https://github.com/bdhammel/faraway-farms)
and used with permission
"""
import numpy as np
from PIL import Image, ImageDraw
from skimage.external import tifffile
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split 
import glob
import os

import utils as pipe_utils


class RawObjImage(pipe_utils.SatelliteImage):
    """
    """

    def __init__(self, image_path):
        self._image_id = pipe_utils.get_file_name_from_path(image_path)

        # import the data for the image and the json
        print("loading image ", self._image_id)
        _data = read_raw_image(image_path)
        self._data = pipe_utils.image_save_preprocessor(_data)
        del _data
        print("...done")
        self._features = {}

    @property
    def features(self):
        return self._features

    def append_feature(self, label, coor):
        """Connect a feature to the imported image

        Args
        ----
        label (str) : label of the feature
        """
        self._features.setdefault(label, []).append(coor)

    def has_labels(self):
        """Return the labels present in this image

        Returns
        -------
        (list : str) : a list of the label names
        """
        return list(self.features.keys())

    def show(self, labels=[], as_poly=True):
        """Display the satellite image with the feature overlay

        Args
        ----
        labels (list) : list of ints corresponding to feature names
        as_poly (bool) : plot the feature as a polygon instead of a bounding box
        """

        labels = pipe_utils.atleast_list(labels)

        im = Image.fromarray(self.data)
        draw = ImageDraw.Draw(im)

        for label in labels:
            locs = self.get_features(as_poly=as_poly)[label]

            for coors in locs:
                if as_poly:
                    draw.polygon(coors, outline='red')
                else:
                    draw.rectangle(coors, outline='red')

        im.show()


def read_raw_image(path, report=True, check_data=False):
    """Import a raw image

    This could be as 8 bit or 16 bit... or 10 bit like some of the files...

    Args
    ----
    path (str) : path to the image file
    report (bool) : output a short log on the imported data

    Raises
    ------
    ImportError : If the image passed does not have a file extension that's expected
    AssertionError : if data_is_ok fails when check_data=True

    Returns
    -------
    numpy array of raw image
    """
    ext = os.path.splitext(path)[1]

    if ext in [".jpg", ".png"]:
        img = imread(path)
    elif ext == ".tif":
        img = tifffile.imread(path)
    else:
        raise ImportError("{} Not a supported extension".format(ext))

    if report:
        print("Image {} loaded".format(os.path.basename(path)))
        print("\tShape: ", img.shape)
        print("\tdtype: ", img.dtype)
        print("Values: ({:.2f},{:.2f})".format(img.min(), img.max())),

    if check_data:
        pipe_utils.data_is_ok(img, raise_exception=True)

    return img


def load_from_categorized_directory(path, load_labels):
    """Load in the raw image files into a dictionary

    The directory should contain folders for each specific class, with the
    relevant images contained inside.

    Args
    ----
    path (str) : the path to the directory where the images are stored

    Returns
    -------
    a dic of the images of form { 'label': [ [img], ...], ...}
    """
    data = {}

    for img_path in glob.glob(path + "/**/*"):

        *_, label, im_name = img_path.split(os.path.sep)

        if label in load_labels:
            try:
                img = read_raw_image(img_path)
            except ImportError:
                # Assuming a file with an extension other than .jpg, .tif,
                # or .png was in the directory, just skip
                #
                # WARNING: this is going on faith that read_raw_image wont
                # raise an error for other reasons... that's probably not a 
                # good assumption
                pass
            else:
                img = pipe_utils.image_save_preprocessor(img)
                data.setdefault(label, []).append(img)

    return data


def generarate_train_and_test(data, path=None, save=False):
    """Take a reduced dataset and make train and test sets

    Warning!
    --------
    Not loading Train and Test sets from files will contaminate your
    Test set with training data

    Args
    ----
    data (dict) : reduced data from convert_classes()
    save (bool) : weather or not to pickle the data

    Returns
    -------
    Xtrain, Xtest, Ytrain, Ytest
    """

    X = []
    Y = []

    if not os.path.exists(path):
        print("Creating directory to save processed images")
        os.makedirs(path)

    for label in data.keys():
        _x = data[label]
        Y += [label] * len(_x)
        X += _x
        del _x

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        np.reshape(X, (-1, 200, 200, 3)),
        np.array(Y),
        test_size=0.33,
        random_state=42
    )

    if save and path is not None:
        pipe_utils.dump_as_pickle(Xtrain, os.path.join(path, "xtrain.p"))
        pipe_utils.dump_as_pickle(Xtest, os.path.join(path, "xtest.p"))
        pipe_utils.dump_as_pickle(Ytrain, os.path.join(path, "ytrain.p"))
        pipe_utils.dump_as_pickle(Ytest, os.path.join(path, "ytest.p"))

    return Xtrain, Xtest, Ytrain, Ytest


def convert_classes(raw_data, local_label_dict):
    """Convert the MC Land use classes to the specific things I'm interested in 

    Args
    ----
    raw_data (dict) : dictionary of raw data, returned from load_raw_data()
    local_label_dict (dict) : a dictionary that maps the datasets labels into 
        the labels used by the model

    Returns
    -------
    Similar dictionary but with labels of specific interest 
    """

    data = {}

    for label, images in raw_data.items():
        local_label = local_label_dict[label]
        if label:
            data[local_label] = images

    return data


def process_patch_for_saving(ds):
    """Resize all images to a formate that works for the network

    Args
    ----
    ds (dict) : 

    Returns
    -------
    (dict)
    """

    clean_ds = {}
    for label, full_images in ds.items():

        for full_image in full_images:
            image = resize(full_image, (200,200), preserve_range=True)
            clean_ds.setdefault(label, []).append(image)

    return clean_ds
