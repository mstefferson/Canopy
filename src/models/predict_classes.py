import os
import imageio
import numpy as np
import rasterio
import src.models.analyze_model
import src.satellite_analyze


class PredImg():
    def __init__(self, imag_str, pix_scale=255., origin_r=0, origin_c=0):
        # paths
        self.imag_path = imag_str
        self.save_path = './results/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # read in image
        self.image = imageio.imread(self.imag_path)
        self.image = self.image / pix_scale
        # set geometry
        self.pix_scale = pix_scale
        self.width = np.shape(self.image)[1]
        self.height = np.shape(self.image)[0]
        self.channels = np.shape(self.image)[2]
        self.geo_scale = [self.width, self.height]
        self.origin = [origin_r, origin_c]
        # labels
        self.num_labels = 1

    def write2file(self, data, fileid):
        filename = self.image_path + fileid
        f = open(filename, 'w+')
        for line in data:
            f.write(str(line) + '\n')

    def writelocalbox2file(self):
        self.write2file(self.boxes, 'bb_local.txt')

    def writeglobalbox2file(self):
        self.write2file(self.boxes_global, 'bb_global.txt')

    def pred_boxes(self):
        self.boxes = np.empty([0, 4])
        self.confidence = np.empty([0, ])
        self.labels = np.empty([0, ])

    def boxes_in_orig(self):
        self.boxes_global = np.ones_like(self.boxes)
        # scale boxes to row columns (and relative to origin)
        self.boxes_global[:, 0] = (
            (self.boxes[:, 0] * self.scale[0]) + self.origin[0])
        self.boxes_global[:, 1] = (
            (self.boxes[:, 1] * self.scale[1]) + self.origin[1])
        self.boxes_global[:, 2] = (self.boxes[:, 2] * self.scale[0])
        self.boxes_global[:, 3] = (self.boxes[:, 3] * self.scale[1])


class PredImgRandom(PredImg):
    def __init__(self, imag_str, pix_scale=255., origin_r=0, origin_c=0):
        PredImg.__init__(self, imag_str, pix_scale=pix_scale,
                         origin_r=origin_r, origin_c=origin_c)

    def pred_boxes(self):
        num_objects = np.random.randint(3)+1
        self.boxes = np.random.rand(num_objects, 4)
        self.confidence = np.random.rand(num_objects,)
        self.labels = np.random.choice(self.num_labels, num_objects)


def divide_tiff(sat_w, sat_h, image_w=200, image_h=200):
    # imagesize = [height, width]
    # get number of divisions
    div_w = int(np.floor(sat_w / image_w))
    div_h = int(np.floor(sat_h / image_h))
    # get trim off edges
    trim_w = sat_w - div_w*image_w
    trim_h = sat_h - div_h*image_h
    # get start indices (columns/width, rows/height)
    start_c = np.floor(trim_w/2)
    end_c = sat_w - np.ceil(trim_w/2)
    start_r = np.floor(trim_h/2)
    end_r = sat_h - np.ceil(trim_h/2)
    # get all origins
    row_origins = np.arange(start_r, end_r, image_w)
    col_origins = np.arange(start_c, end_c, image_h)
    return row_origins, col_origins



def pred_subset(sat_data, r_start, r_end, c_start, c_end,
                model='none', c_channels=[0, 1, 3]):
    # check to make sure it's not zero
    delta_r = r_end - r_start
    delta_c = c_end - c_start
    zero_compare = np.zeros((delta_r, delta_c, 3))
    # get band data
    band_data = src.satellite_analyze.get_satellite_subset(
        sat_data, r_start, r_end, c_start, c_end)
    band_data = band_data[:, :, c_channels]
    if np.all(band_data != zero_compare):
        predictions = np.random.rand(np.random.randint(3)+1, 5)
    else:
        predictions = None
    return predictions


def predict(model, data):
    # get boxes 
    box_list, bboxes = predict_bounding_box(yolo, image, config['model']['labels'])
    if write_file:
        f = open('results.txt', 'w+')
        for box in box_list:
            f.write(str(box) + '\n')
    if  save_detect:
        image = draw_boxes(image, bboxes, config['model']['labels'])
        filename = image_path[:-4] + '_detected' + image_path[-4:]
        cv2.imwrite(filename, image)


def pred_tiff(sat_file,  r_start=0,
              r_end=np.inf, c_start=0, c_end=np.inf,
              image_w=200, image_h=200):
    # get data
    sat_data = rasterio.open(sat_file)
    sat_width = sat_data.width
    sat_height = sat_data.height
    scale_image = [image_h, image_w]
    # get orgins for each subset
    r_end = np.min([r_end, sat_data.height])
    c_end = np.min([c_end, sat_data.width])
    # get orgins
    origins_r, origins_c = divide_tiff(sat_width, sat_height,
                                       image_w=200, image_h=200)
    origins_r = origins_r[(origins_r > r_start) &
                          (origins_r < r_end)].astype(int)
    origins_c = origins_c[(origins_c > c_start) &
                          (origins_c < c_end)].astype(int)
    origin_list = [(r, c) for r in origins_r for c in origins_c]
    print('sat file ({}x{})'.format(sat_height, sat_width))
    print('Number of origins', len(origin_list))
    counter = 0
    # keep a list of all located trees
    tree_dict_info = []
    tree_info = np.empty((0, 4), int)
    for origin in origin_list:
        pred = pred_subset(sat_data,
                           origin[0], origin[0]+image_h,
                           origin[1], origin[1]+image_w)
        # make a list of all located trees
        pred_dict = {}
        pred_dict['origin'] = origin
        pred_dict['size'] = scale_image
        pred_dict['local'] = pred
        # put local prediction into global image
        if pred is not None:
            # convert to global
            pred_global = np.ones_like(pred)
            pred_global[:, 0] = (pred[:, 0] * scale_image[0]) + origin[0]
            pred_global[:, 1] = (pred[:, 1] * scale_image[1]) + origin[1]
            pred_global[:, 2] = (pred[:, 2] * scale_image[0])
            pred_global[:, 3] = (pred[:, 3] * scale_image[1])
            pred_global[:, 4] = pred[:, 4]
            # convert to int
            pred_dict['global'] = pred_global
            tree_info = np.append(tree_info, pred_global[:, :4].astype('int'),
                                  axis=0)
        else:
            pred_dict['global'] = None
        tree_dict_info.append(pred_dict)
        counter += 1
        if np.mod(counter, np.floor(len(origin_list)/20)) == 0:
            print('Percent done:', counter / len(origin_list))
    return tree_dict_info, tree_info
