"""Functions specific to handling cleaned data for the Object detection model.
Used to visualize labels.

Code written by Ben Hammel (https://github.com/bdhammel/faraway-farms)
and used with permission
"""

import csv
import os
from PIL import Image, ImageDraw

import utils
import data_utils


class ObjImage(utils.SatelliteImage):
    """Class to handle images with bounding box attributes

    ObjImage inherits from Satellite Image to include the generic properties:

    Parameters
    ----------
    _data (np.array, int) : image data of the type uint8 with range [0,255]
    _image_id (str) : a unique identifier for the image, if the image is saves
    this becomes the file name
    _features ( {label: [(int, int, int, int), ...], ...}) : (optional)
        object features in the image, this property isn't used with patch
        images
    """

    def __init__(self, image_path=None, data=None, *args, **kwargs):
        """Loads an image from a processed directory

        Args
        ----
        image_path (str) : location of the image to upload
        data (np.array) : raw data to load. Data is loaded this way in the
        last stage of cleaning, to ensure that internal tests pass
        """

        if image_path is not None:
            _data = data_utils.read_raw_image(image_path)
            _image_id = utils.get_file_name_from_path(image_path)
        elif data is not None:
            _data = data
            _image_id = None

        super().__init__(data=_data, image_id=_image_id, use='obj',
                         *args, **kwargs)

    def get_features(self):
        """Return all the features associated with the image

        Returns
        -------
        dict {label : [(x1, y1, x2, y2), (x1, y1, x2, y2), ...], ...}
        """
        return self._features

    def append_feature(self, label, coor):
        """Connect a feature to the imported image

        Args
        ----
        label (str) : label of the feature
        coor (tuple : int) : coordinates of the bbox, of form (x1, y1, x2, y2)
        """
        self._features.setdefault(label, []).append(coor)

    def has_labels(self):
        """Return the labels connected to this image

        Returns
        --------
        [label1, label2, ...]
        """
        return list(self._features.keys())

    def show(self, labels=None, return_image=False):
        """Display the image

        Args
        ----
        labels (str) : label to be plotted with the image, can be str, list,
            or 'all' to plot all labels in the image
        return_image (bool) : don't plot the image, just return it, used for
            plotting inline with jupyter notebooks
        """

        im = Image.fromarray(self.data)

        # If 'all' get all labels in the image
        if labels == 'all':
            labels = self.has_labels()

        # Make sure labels is a list
        labels = utils.atleast_list(labels)

        # Initialize a drawer, and draw each feature
        draw = ImageDraw.Draw(im)

        for label in labels:
            locs = self.get_features()[label]

            for coors in locs:
                draw.rectangle(
                    coors,
                    outline='red'
                )

        if return_image:
            return im
        else:
            im.show()


def load_data(annotations_file, max_images=100):
    """Load data in from a CSV annotations file

    Args
    ----
    annotations_file (str) : the path to the annotations.csv file
    max_images (int) : the maximum number of images to load from a given
    annotations_file
    """

    dataset = {}

    with open(annotations_file) as f:
        csv_reader = csv.reader(f)

        for img_path, *coor, label in csv_reader:

            # Convert coor to ints (pixels), if no coor, then just pass
            try:
                coor = tuple(map(float, coor))
            except ValueError as e:
                pass
            else:
                img = dataset.get(img_path, ObjImage(img_path))
                img.append_feature(label, coor)
                dataset[img_path] = img

            if len(dataset.keys()) >= max_images:
                break

    return list(dataset.values())


if __name__ == '__main__':
    data_path = os.getcwd() + "/data/processed/dstl/annotations/"
    ds = load_data(data_path + 'annotations.csv')

    for data in ds:
        if 'trees' in data.has_labels():
            data.show(labels='trees')
            input("press enter to continue")
