"""Base functions used to handle clean data before it's feed into a model
"""
import numpy as np
from skimage.util import view_as_blocks
from skimage.color import grey2rgb
from PIL import Image
import pickle
import math
import os


# Conversion of labels to id for path classification
PATCH_CLASS_TO_ID = {
    'trees': 0,
    'water': 1,
    'crops': 2,
    'vehicles': 3,
    'buildings': 4,
    'field': 5
}

# Conversion of labels to id for object detection
OBJ_CLASS_TO_ID = {
    'vehicles': 0,
    'buildings': 1,
    'animals': 2,
    'trees': 3
}


class SatelliteImage:
    """Base class to store images for loading into the models

    Properties
    ----------
    _data (np.array) : image data
    _features ( {label: [(int, int, int, int), ...], ...}) : (optional)
        object features in the image, this property isn't used with patch images
    _image_id (str) : a identifier to describe the image
    """

    def __init__(self, data, image_id=None, use=None, *args, **kwargs):
        """
        Args
        ----
        data (np array) : (optional) image data, typical this is set in the
            inherited instances
        use (str) : ['obj', 'patch'] The child instance that is loading data,
            this determines what tests are run on the data
        """
        self._data = data
        self._image_id = image_id

        # initialize a blank dictionary to store
        self._features = {}

        # run internal test to make sure data wasn't loaded incorrectly,
        # or from a dirty file
        data_is_ok(self._data, use, *args, **kwargs)

    @property
    def data(self):
        """Return the satellite image data
        """
        return self._data

    @property
    def image_id(self):
        """Return the id of the image, this is usually the file name

        Raises
        ------
        Exception : if no id has been assigned
        """
        if self._image_id is None:
            raise Exception("No id for this image")

        return self._image_id

    def set_image_id(self, image_id):
        """Give this data an id

        Note
        ----
        Care should be taken to ensure this is a unique id, but there is no
        test in place to check this. Non-Unique ids could result in multiple
        images being saved with the same name

        Args
        ----
        image_id (str) : the id to assign the image
        """
        self._image_id = image_id

    def show(self):
        """Display the image

        Typically this function if overloaded

        Returns
        -------
        (PIL.IMAGE)
        """
        im = Image.fromarray(self.data)
        im.show()
        return im


def ids_to_classes(ids, use):
    """Convert id integers back to a verbose label

    Args
    ----
    ids ( np.array ) : numpy array of ids to convert to verbose labels
    use (str) : 'patch' or 'obj' what

    Returns
    -------
    [label, label, ...]
    """
    if use == 'patch':
        class_items = PATCH_CLASS_TO_ID.items()
    elif use == 'obj':
        class_items = OBJ_CLASS_TO_ID.items()
    else:
        raise Exception("invalid use")

    labels = []
    for _id in ids:
        for key, value in class_items:
            if value == _id:
                labels.append(key)

    return labels


def dump_as_pickle(data, path):
    """Save a given python object as a pickle file

    save an object as a "obj.p"

    Args
    ----
    data (python object) : the object to save
    path (str) : the location to save the data, including the file name w/
        extension
    """

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickled_data(path):
    """Load in a pickled data file

    Args
    ----
    path (str) : path to the file to read

    Returns
    -------
    the data object
    """

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def chop_to_blocks(data, shape):
    """Subdivides the current image and returns an array of images
    with the dims `shape`

    Args
    ----
    shape (tuple : ints) : the dims of the subdivided images

    Returns
    -------
    matrix of (j, i, 1, shape[0], shape[1])
    """

    # Make sure there are not multiple strides in the color ch direction
    assert shape[-1] == data.shape[-1]

    # Drop parts of the image that cant be captured by an integer number of
    # strides
    _split_factor = np.floor(np.divide(data.shape, shape)).astype(int)
    _img_lims = (_split_factor * shape)

    # print("Can only preserve up to pix: ", _img_lims)

    _data = np.ascontiguousarray(
        data[:_img_lims[0], :_img_lims[1], :_img_lims[2]]
    )

    return view_as_blocks(_data, shape)


def as_batch(img, shape, as_list=False):
    """Convert an image block group to a list

    After chop_to_blocks, the data has a structure like (jx, ix, 1, 400, 400, 3)
    convert this to:

    obj_detection : (ix*jx, 400, 400, 3)
    patch_identification : (ix*jx, 200, 200, 3)

    Args
    ----
    img ( np.array) : image data
    as_list (bool) : return as a list and not a numpy array
    """

    blocks = chop_to_blocks(img, shape=shape)
    og_shape = blocks.shape

    flat_blocks = blocks.reshape(np.prod(og_shape[:3]), *og_shape[3:])

    if as_list:
        return list(flat_blocks)
    else:
        return flat_blocks


def get_file_name_from_path(path):
    """Extract the file name from a given path

    If path is `/path/to/file/name.ext` then this functions returns `name`

    Args
    ----
    path (str) : path to a file of interest

    Returns
    -------
    (str) The file name
    """
    return os.path.splitext(os.path.basename(path))[0]


def atleast_list(thing):
    """Make sure the item is at least a list of len(1) if not a list
    otherwise, return the original list

    Args
    ----
    thing (any type) : thing to assert is a list

    Returns
    -------
    thing (list)
    """
    if not isinstance(thing, list):
        thing = [thing]

    return thing


def data_is_ok(data, use=None, raise_exception=False):
    """Perform a check to ensure the image data is in the correct range

    Args
    ----
    data (np.array) : the image data
    use (str) : ['obj', 'patch', 'None'] the type (or use) of image passed,
        this is for checking the image shape, if None, don't check the shape
    raise_exception (bool) : raise exception if data is not ok

    Returns
    -------
    (bool) : True if data is OK, otherwise False
    """
    try:
        assert data.dtype == np.uint8
        assert data.max() <= 255
        assert data.min() >= 0

        # make sure data wasn't normalized to [0,1)
        assert data.max() > 1.0

        if use == 'obj':
            assert data.shape == (400, 400, 3)
        elif use == 'patch':
            assert data.shape == (200, 200, 3)
    except AssertionError as e:
        if raise_exception:
            raise e
        else:
            _data_is_ok = False
    else:
        _data_is_ok = True

    return _data_is_ok


def image_save_preprocessor(img, report=True):
    """Normalize the image

    Procedure
    ---------

     - Convert higher bit images (16, 10, etc) to 8 bit
     - Set color channel to the last channel
     - Drop Alpha layer and conver b+w -> RGB

    TODO
    ----
    Correctly handle images with values [0,1)

    Args
    ----
    img (np array) : raw image data
    report (bool) : output a short log on the imported data

    Returns
    -------
    numpy array of cleaned image data with values [0, 255]
    """

    data = np.asarray(img)

    if data.ndim == 3:
        # set the color channel to last if in channel_first format
        if data.shape[0] <= 4:
            data = np.rollaxis(data, 0, 3)

        # remove alpha channel
        if data.shape[-1] == 4:
            data = data[..., :3]

    # Convert to color if B+W
    if len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[-1] == 1):
        data = grey2rgb(data)

    # if > 8 bit, shift to a 255 pixel max
    bitspersample = int(math.ceil(math.log(data.max(), 2)))
    if bitspersample > 8:
        data >>= bitspersample - 8

    # if data [0, 1), then set range to [0,255]
    if bitspersample <= 0:
        data *= 255

    data = data.astype(np.uint8)

    if report:
        print("Cleaned To:")
        print("\tShape: ", data.shape)
        print("\tdtype: ", data.dtype)

    # Make sure the data is actually in the correct format
    data_is_ok(data, raise_exception=True)

    return data
