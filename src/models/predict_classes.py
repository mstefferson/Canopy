import os
import imageio
import numpy as np
import rasterio
import src.models.analyze_model
import src.satellite_analyze


class PredImg():
    def __init__(self, num_labels=1, origin_r=0, origin_c=0):
        # paths
        self.imag_path = imag_str
        self.save_path = './results/'
        self.origin = [origin_r, origin_c]
        # labels
        self.num_labels = num_labels
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def set_geometry(self):
        # set the geometry
        self.width = np.shape(self.image)[1]
        self.height = np.shape(self.image)[0]
        self.channels = np.shape(self.image)[2]
        self.geo_scale = [self.width, self.height]

    def load_imag_from_path(self, imag_path):
        # read in image
        self.image = imageio.imread(imag_path)

    def set_image(self, image):
        # set image
        self.image = image

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
        self.boxes = np.empty([0, 6])

    def get_boxes_in_orig(self):
        self.boxes_global = np.ones_like(self.boxes)
        # scale boxes to row columns (and relative to origin)
        self.boxes_global[:, 1] = (
            (self.boxes[:, 1] * self.scale[1]) + self.origin[0])
        self.boxes_global[:, 2] = (
            (self.boxes[:, 2] * self.scale[2]) + self.origin[1])
        self.boxes_global[:, 3] = (self.boxes[:, 3] * self.scale[0])
        self.boxes_global[:, 4] = (self.boxes[:, 4] * self.scale[1])


class PredImgRandom(PredImg):
    def __init__(self, imag_str, pix_scale=255., origin_r=0, origin_c=0):
        PredImg.__init__(self, origin_r=origin_r, origin_c=origin_c)

    def pred_boxes(self):
        num_objects = np.random.randint(3)+1
        self.boxes = np.random.rand(num_objects, 4)
        self.confidence = np.random.rand(num_objects,)
        self.labels = np.random.choice(self.num_labels, num_objects)


class SatelliteTif():
    def __init__(self, tif_path, pred_class,
                 c_channels=[0, 1, 3], sub_img_w=200, sub_img_h=200,
                 r_start=0, r_end=np.inf, c_start=0, c_end=np.inf):
        # store tif
        self.tiff_data = rasterio.open(tif_path)
        # store geometry conversion type
        self.crs = self.tiff_data.crs
        # lat/lon project string
        self.lonlat_proj = 'epsg:4326'
        # set geometry
        self.tif_norm = 65535.
        self.jpg_norm = 355
        self.c_channels = c_channels
        self.sat_full_w = self.tiff_data.width
        self.sat_full_h = self.tiff_data.height
        self.img_w = self.sub_img_w
        self.img_h = self.sub_img_h
        # set prediction class
        self.pred_class = pred_class
        # get orgins for each subset
        self.r_end = np.min([r_end, self.sat_h])
        self.r_start = r_start
        self.c_end = np.min([c_end, self.sat_w])
        self.c_start = c_start
        self.sat_pred_w = self.c_end - self.c_start
        self.sat_pred_h = self.r_end - self.r_start
        # get origins
        self.build_origins()
        # initialize a subset to self
        self.data = np.zeros(self.image_h, self.image_w, 3)
        # all objects
        self.detected_objects = np.empty([0, 6])

    def proj_lonlat_2_rc(self, lonlat):
        '''
        Description:
            Convert lon/lat coordinates to rows and columns in the tif satellite
            image. Uses pyproj to convert between coordinate systems
        Args:
            lonlat (np.array, size=[n,2]): array of longitude (col1) and lat (col2)
        Returns:
            rc (np.array shape=[n, 2]): row/columns in tif file for all
                recorded points
        Updates:
            N/A
        Write to file:
            N/A
        '''
        # input lat/lon
        in_proj = pyproj.Proj(init=self.lonlat_proj)
        # output based on crs of tif/shp
        out_proj = pyproj.Proj(dataset.crs)
        # transform lat/lon to xy
        x, y = pyproj.transform(in_proj, out_proj, lonlat[:, 0], lonlat[:, 1])
        # convert rows and columns to xy map project
        (r, c) = rasterio.transform.rowcol(dataset.transform, x, y)
        # store it in numpy array
        rc = np.array([r, c]).transpose()
        return rc


    def proj_rc_2_lonlat(self, rc):
        '''
        Description:
            Convert row/columns of tif dataset to lat/lon.
            Uses pyproj to convert between coordinate systems
        Args:
            lon (float): longitude
            lat (float): latitude
            dataset (rasterio.io.DatasetReader): Gdal data structure from opening a
                tif, dataset = rasterio.open('...')
        Returns:
            rc (np.array shape=[n, 2]): row/columns in tif file for all
                recorded points
        Updates:
            N/A
        Write to file:
            N/A
        '''
        # convert rows and columns to xy map project
        (x, y) = rasterio.transform.xy(dataset.transform, rc[:, 0], rc[:, 1])
        # input based on crs of tif/shp
        in_proj = pyproj.Proj(self.crs)
        # output lat/lon
        out_proj = pyproj.Proj(init=self.lonlat_proj)
        # transform xy to lat/lon
        lon, lat = pyproj.transform(in_proj, out_proj, x, y)
        # store it in numpy array
        lonlat = np.array([lon, lat]).transpose()
        return lonlat

    def build_origins(self):
        # get number of divisions
        div_w = int(np.floor(self.sat_pred_w / self.img_w))
        div_h = int(np.floor(self.sat_pred_h / self.img_h))
        # get trim off edges
        trim_w = self.sat_pred_w - div_w*image_w
        trim_h = self.sat_pred_h - div_h*image_h
        # fix start indices (columns/width, rows/height)
        self.start_c = self.start_c + np.floor(trim_w/2)
        self.end_c = self.end_c - np.ceil(trim_w/2)
        self.start_r = self.start_c + np.floor(trim_h/2)
        self.end_r = self.end_r - np.ceil(trim_h/2)
        # get all origins
        row_origins = np.arange(self.start_r, self.end_r, self.img_w)
        col_origins = np.arange(self.start_c, self.end_c, self.img_r)
        self.origin_list = [(r, c) for r in row_origins for c in col_origins]

    def divide_tif():
        print('write me')

    def get_subset(self, r_start, r_end, c_start, c_end):
        for (c_id, c) in enumerate(self.c_channels):
            self.data[:, :, c_id] = ds_all.read(band_num,
                                                window=((r_start, r_end),
                                                        (c_start, c_end)))
        # normalize
        self.data = self.data / self.tif_norm * self.jpg_norm

    def pred_subset(self, r_start, r_end, c_start, c_end):
        self.data = self.get_subset(r_start, r_end, c_start, c_end)
        # set up prediction object
        self.pred_class(origin_r=r_start, origin_c=c_start)
        boxes = self.pred_class.pred_boxes(self.data)
        self.pred_class.boxes_in_orig()
        return self.pred_class.boxes_global

    def pred_all(self):
        for origin in self.origin_list:
            boxes = pred_subset(sat_data,
                                origin[0], origin[0]+self.img_h,
                                origin[1], origin[1]+self.img_w)
            self.boxes_global = np.append(self.boxes_global, boxes)

# def pred_tiff(sat_file,  r_start=0,
              # r_end=np.inf, c_start=0, c_end=np.inf,
              # image_w=200, image_h=200):
    # # get data
    # sat_data = rasterio.open(sat_file)
    # sat_width = sat_data.width
    # sat_height = sat_data.height
    # scale_image = [image_h, image_w]
    # # get orgins for each subset
    # r_end = np.min([r_end, sat_data.height])
    # c_end = np.min([c_end, sat_data.width])
    # # get orgins
    # origins_r, origins_c = divide_tiff(sat_width, sat_height,
                                       # image_w=200, image_h=200)
    # origins_r = origins_r[(origins_r > r_start) &
                          # (origins_r < r_end)].astype(int)
    # origins_c = origins_c[(origins_c > c_start) &
                          # (origins_c < c_end)].astype(int)
    # origin_list = [(r, c) for r in origins_r for c in origins_c]
    # print('sat file ({}x{})'.format(sat_height, sat_width))
    # print('Number of origins', len(origin_list))
    # counter = 0
    # # keep a list of all located trees
    # tree_dict_info = []
    # tree_info = np.empty((0, 4), int)
    # for origin in origin_list:
        # pred = pred_subset(sat_data,
                           # origin[0], origin[0]+image_h,
                           # origin[1], origin[1]+image_w)
        # # make a list of all located trees
        # pred_dict = {}
        # pred_dict['origin'] = origin
        # pred_dict['size'] = scale_image
        # pred_dict['local'] = pred
        # # put local prediction into global image
        # if pred is not None:
            # # convert to global
            # pred_global = np.ones_like(pred)
            # pred_global[:, 0] = (pred[:, 0] * scale_image[0]) + origin[0]
            # pred_global[:, 1] = (pred[:, 1] * scale_image[1]) + origin[1]
            # pred_global[:, 2] = (pred[:, 2] * scale_image[0])
            # pred_global[:, 3] = (pred[:, 3] * scale_image[1])
            # pred_global[:, 4] = pred[:, 4]
            # # convert to int
            # pred_dict['global'] = pred_global
            # tree_info = np.append(tree_info, pred_global[:, :4].astype('int'),
                                  # axis=0)
        # else:
            # pred_dict['global'] = None
        # tree_dict_info.append(pred_dict)
        # counter += 1
        # if np.mod(counter, np.floor(len(origin_list)/20)) == 0:
            # print('Percent done:', counter / len(origin_list))
    # return tree_dict_info, tree_info
