import os
import imageio
import numpy as np
import rasterio
import src.models.analyze_model


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
    def __init__(self, tif_path, rel_path_2_data,
                 c_channels=[0, 1, 3], sub_img_w=200, sub_img_h=200,
                 train_window=0.4, num_train=100, valid_frac=0.3,
                 r_start=0, r_end=np.inf, c_start=0, c_end=np.inf):
        # store tif
        self.sat_data = rasterio.open(tif_path)
        # store geometry conversion type
        self.crs = self.sat_data.crs
        # lat/lon project string
        self.lonlat_proj = 'epsg:4326'
        # set geometry
        self.tif_norm = 65535.
        self.jpg_norm = 355
        self.c_channels = c_channels
        self.sat_w = self.sat_data.width
        self.sat_h = self.sat_data.height
        self.img_w = sub_img_w
        self.img_h = sub_img_h
        # save directory
        leading_zeros = int(np.ceil(np.log10(np.max([self.sat_w,
                                                     self.sat_h]))))
        self.image_save_format = (
            'image_{:0' + str(leading_zeros) + 'd}' +
            '_{:0' + str(leading_zeros) + 'd}')
        self.base_dir = os.getcwd() + '/' + rel_path_2_data
        self.pred_dir = self.base_dir + '/predict'
        self.train_dir = self.base_dir + '/train'
        self.valid_dir = self.base_dir + '/valid'
        self.build_directories()
        # set prediction class and train params
        self.valid_frac = valid_frac
        # get orgins for each subset
        self.r_end = np.min([r_end, self.sat_h])
        self.r_start = r_start
        self.c_end = np.min([c_end, self.sat_w])
        self.c_start = c_start
        self.pred_origins = self.build_origins(self.r_start,
                                               self.r_end,
                                               self.c_start,
                                               self.c_end)
        # got training, sample from some fraction of the center of image
        delta_window_start = (1-train_window) / 2
        delta_window_end = 1 - (1-train_window) / 2
        poss_train_origins = self.build_origins(delta_window_start*self.sat_h,
                                                delta_window_end*self.sat_h,
                                                delta_window_start*self.sat_w,
                                                delta_window_end*self.sat_w)
        # get random sample
        max_train_num = len(poss_train_origins)
        num_train = np.min([num_train, max_train_num])
        self.num_all = num_train
        self.num_valid = int(self.valid_frac * num_train)
        self.num_train = self.num_all - self.num_valid
        idx = np.random.choice(np.arange(max_train_num),
                               size=self.num_all, replace=False)
        self.training_origins = poss_train_origins[idx, :]
        self.train_origins = self.training_origins[:self.num_train, :]
        self.valid_origins = self.training_origins[self.num_train:, :]

    def build_directories(self):
        dirs2build = [self.base_dir, self.pred_dir,
                      self.train_dir, self.valid_dir]
        for a_dir in dirs2build:
            # build dirs
            if not os.path.exists(a_dir):
                os.makedirs(a_dir)
                # build image dirs
                if not a_dir == self.base_dir:
                    image_dir = a_dir+'/images'
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    # build label dirs
                    if a_dir in [self.train_dir, self.valid_dir]:
                        label_dir = a_dir+'/labels'
                        if not os.path.exists(label_dir):
                            os.makedirs(label_dir)


    def proj_lonlat_2_rc(self, lonlat):
        '''
        Description:
            Convert lon/lat coordinates to rows and columns in the tif
            satellite image. Uses pyproj to convert between coordinate systems
        Args:
            lonlat (np.array, size=[n,2]): array of longitude
                (col1) and lat (col2)
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
            rc (np.array shape=[n, 2]): row/columns in tif file for all
                recorded points
        Returns:
            lonlat (np.array, size=[n,2]): array of longitude (col1) and lat (col2)
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

    def build_origins(self, start_r, end_r, start_c, end_c):
        # get number of divisions
        div_w = int(np.floor(self.sat_w / self.img_w))
        div_h = int(np.floor(self.sat_h / self.img_h))
        # set start and end to start/end on a division point
        start_c = np.floor(start_c/div_w) * div_w
        end_c = np.floor(end_c/div_w) * div_w
        start_r = np.floor(start_r/div_h) * div_w
        end_r = np.floor(end_r/div_h) * div_w
        # get all origins
        row_origins = np.arange(start_r, end_r, self.img_h)
        col_origins = np.arange(start_c, end_c, self.img_w)
        origin_list = np.array([(r, c) for r in row_origins
                                for c in col_origins])
        return origin_list

    def build_train_dataset():
        self.build_dataset(self.train_origins, self.train_dir)

    def build_valid_dataset():
        self.build_dataset(self.valid_origins, self.valid_dir)

    def build_pred_dataset():
        self.build_dataset(self.pred_origins, self.pred_dir)

    def build_dataset(self, origins, save_dir):
        # build full array (r_start, r_end, c_start, c_end)
        all_coors = np.zeros((len(origins, 4)))
        all_coors[:, 0] = origins[:, 0]
        all_coors[:, 1] = origins[:, 0] + delta
        all_coors[:, 2] = origins[:, 1]
        all_coors[:, 3] = origins[:, 1] + delta
        # get key name from save dir
        name_str = '/'.splirt(save_dir)[-1]
        with open(self.base_dir + name_str + '_key.txt', 'w+') as look_up_f:
            for origin in origins:
                # set image save name
                img_save_id = self.image_save_format.format(origin[0],
                                                            origin[1])
                # get subset
                band_data = self.get_subset(coors[ii, 0],
                                            coors[ii, 1],
                                            coors[ii, 2],
                                            coors[ii, 3])
                # update band to save based on color channels
                # must be saved as integars from 0, 255
                band2save = np.array(band_data, dtype=np.uint8)
                # save everything
                img_save_name = image_save_id + '.jpg'
                # same image
                imageio.imwrite(save_dir + img_save_name, band2save)
                # update lookup dict
                look_up_dict[img_save_name] = train_coors[ii, :]
                # update lookup txt file
                line2save = (img_save_name
                             + ' ' + str(coors[ii, 0])
                             + ' ' + str(coors[ii, 1])
                             + ' ' + str(coors[ii, 2])
                             + ' ' + str(coors[ii, 3])
                             + '\n')
                look_up_f.write(line2save)

    def build_test_train(sat_file, num_images, delta=200,
                         split=0.3, c_channels=[0, 1, 3], fid_start=0):
        '''
        Description:
            Builds a test/train set from random squares of a sat_file tif file
        Args:
            sat_file (str): path to satellite tif file
            num_images (int): total number of images in data set
            delta (int, optional): width/heigth of image in data set
            split (float, optional): test/train split (frac in test)
            c_channels (list of ints, optional): three color bands
                to include in image. Tiff has four: r, g, b, IR
            fid_start: initial file id to label images. e.g., image_fid.jpg
        Returns:
            N/A
        Updates:
            N/A
        Write to file:
            /app/data/(test,train)/images/image_*,
            /app/data/(test,train)/images/key*: test and train image data
                set with keys

        '''
        def save_data(sat_data, coors, num2save, num_total, path, save_str, fid):
            '''
            Description:
                Nested function that actually writes the files
            '''
            look_up_name = 'key'
            look_up_f = open(path + look_up_name + '.txt', 'w+')
            look_up_dict = {}
            for ii in np.arange(num2save):
                # get subset
                band_data = get_satellite_subset(sat_data,
                                                 coors[ii, 0],
                                                 coors[ii, 1],
                                                 coors[ii, 2],
                                                 coors[ii, 3])
                # update band to save based on color channels
                # must be saved as integars from 0, 255
                band2save = np.array(255*band_data[:, :, c_channels],
                                     dtype=np.uint8)
                # save everything
                img_save_name = save_str.format(fid+ii) + '.jpg'
                # same image
                imageio.imwrite(path + img_save_name, band2save)
                # update lookup dict
                look_up_dict[img_save_name] = train_coors[ii, :]
                # update lookup txt file
                line2save = (img_save_name
                             + ' ' + str(coors[ii, 0])
                             + ' ' + str(coors[ii, 1])
                             + ' ' + str(coors[ii, 2])
                             + ' ' + str(coors[ii, 3])
                             + '\n')
                look_up_f.write(line2save)
            # save lookup dictionary
            pickle.dump(look_up_dict, open(path + look_up_name + '.pkl', 'wb'))
            # close txt file
            look_up_f.close()
        # build output directories if they don't exist
        directory = '/app/data/athens/train/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = '/app/data/athens/test/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # get sat data
        sat_data = rasterio.open(sat_file)
        # get number of rows and columns
        num_r = sat_data.height
        num_c = sat_data.width
        # find all possible subsections
        r_min = num_r / 4
        r_max = 3*num_r / 4
        c_min = num_c / 4
        c_max = 3*num_c / 4
        # get all possible row/column starting points
        r_start_all = np.arange(r_min, r_max, delta)
        c_start_all = np.arange(r_min, r_max, delta)
        # get random sample
        r_start = np.random.choice(r_start_all, num_images)
        c_start = np.random.choice(c_start_all, num_images)
        # coordinates
        start_coord = np.array(
            [r_start, c_start]).transpose()[:num_images].astype('int')
        # build full array (r_start, r_end, c_start, c_end)
        all_coors = np.zeros((num_images, 4))
        all_coors[:, 0] = start_coord[:, 0]
        all_coors[:, 1] = start_coord[:, 0] + delta
        all_coors[:, 2] = start_coord[:, 1]
        all_coors[:, 3] = start_coord[:, 1] + delta
        # split into test/train
        all_coors = all_coors.astype('int')
        num_test = int(np.floor(num_images * split))
        num_train = int(num_images - num_test)
        train_coors = all_coors[:num_train, :]
        test_coors = all_coors[num_train:, :]
        # set str format for save name
        num_dec = int(np.ceil(np.log10(fid + num_total))) + 1
        save_str = 'image_{:0' + str(num_dec) + 'd}'
        # save train
        train_path = '/app/data/train/images/'
        save_data(sat_data, train_coors, num_train,
                  num_images, train_path, save_str, fid)
        # save test
        test_path = '/app/data/test/images/'
        save_data(sat_data, test_coors, num_test,
                  num_images, test_path, save_str, fid+num_train)

    def divide_tif():
        print('write me')

    def get_subset(self, r_start, r_end, c_start, c_end):
        for (c_id, c) in enumerate(self.c_channels):
            self.data[:, :, c_id] = self.sat_data.read(band_num,
                                                       window=((r_start, r_end)
                                                               (c_start, c_end)))
        # normalize
        self.data = np.array(self.data / self.tif_norm * self.jpg_norm,
                             dtype=np.uint8)

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
