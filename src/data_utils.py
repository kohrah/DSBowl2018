import numpy as np
import pandas as pd
import os


if os.name == 'nt':
    GLOBAL_PATH = 'E:\Datasets\Spacenet\\train\\'
else:
    GLOBAL_PATH = '/home/nosok/datasets/SpaceNet/train'

class SpaceNetDataset(object):
    def __init__(self, path=None, train_fraction=0.8, seed=None):
        if path == None:
            self.path = GLOBAL_PATH
        else:
            self.path = path
        self.pan_subpath = 'PAN\\'
        self.rgb_pansharpen_subpath = 'RGB-PanSharpen\\'
        self.json_subpath = 'geojson\\spacenetroads\\'
        self.cities_list = ['AOI_2_Vegas_Roads_Train', 'AOI_3_Paris_Roads_Train',
               'AOI_4_Shanghai_Roads_Train', 'AOI_5_Khartoum_Roads_Train']
        self.train_fraction = train_fraction
        self.all_data = self.get_files_ids()
        if seed == None:
            self.seed = 1
        else:
            self.seed = seed
        self.train_data = self.splitter(is_train=True)
        self.val_data = self.splitter(is_train=False)

    def get_files_ids(self):
        files_list = []
        for city in self.cities_list:
            path = os.path.join(self.path, city, 'summaryData', city+'.csv')
            tmp_df = pd.read_csv(path)#GLOBAL_PATH + city + '\\summaryData\\' + city + '.csv')
            tmp_df = tmp_df[tmp_df['WKT_Pix'] != 'LINESTRING EMPTY']
            tmp_list = list(set(tmp_df['ImageId']))
            for item in tmp_list:
                files_list.append(item)
        return files_list

    def check_is_dividable(self, data):
        if len(data) % 2 == 0:
            return data
        else:
            data.append(data[0])
            return data

    def splitter(self, is_train):
        import random
        random.seed(self.seed)
        random.shuffle(self.all_data)
        split_at = int(np.ceil(len(self.all_data) * self.train_fraction))
        if is_train:
            return self.check_is_dividable(self.all_data[:split_at])
        else:
            return self.check_is_dividable(self.all_data[split_at:])


import cv2
import tifffile
import os
import random
from keras.preprocessing.image import Iterator

class SpaceNetIterator(Iterator):
    def __init__(self, lock, dataset, batch_size, use_weights=False, weight=5, is_train=False, datagen=None, unet_size=512,
                 shuffle=True, path=None, seed=None, img_type='RGB-PanSharpen'):
        self.cities_list = ['AOI_2_Vegas_Roads_Train', 'AOI_3_Paris_Roads_Train',
               'AOI_4_Shanghai_Roads_Train', 'AOI_5_Khartoum_Roads_Train']
        self.dataset = dataset
        self.len = len(self.dataset)
        self.datagen = datagen
        self.batch_size = batch_size
        self.use_weights = use_weights
        self.weight = weight
        self.is_train = is_train
        self.img_type = img_type
        self.img_size = unet_size
        self.shuffle = shuffle
        if path is None:
            self.path = GLOBAL_PATH
        else:
            self.path = path
        self.df = self.load_dfs()
        if seed is None:
            self.seed = 1
        else:
            self.seed = seed
        super(SpaceNetIterator, self).__init__(self.len, self.batch_size, shuffle, seed)
        self.lock = lock

    def load_dfs(self):
        print('reading df...')
        df = pd.DataFrame(columns=['ImageId', 'WKT_Pix'])
        for city in self.cities_list:
            df_path = os.path.join(self.path, city, 'summaryData', city+'.csv')
            tmp_df = pd.read_csv(df_path, names=['ImageId', 'WKT_Pix'])
            df = df.append(tmp_df)
        del tmp_df
        #df = df[df['WKT_Pix'] != 'LINESTRING EMPTY']
        return df

    def get_path(self, r):
        s = r.rsplit('_i')[:-1]
        im_path = os.path.join(self.path, s[0] + '_Roads_Train', self.img_type, r + '.tif')
        return im_path

    def split_row(self, r):
        return r.rsplit(',')

    def weird_shit(self, okay):
        # FIXME. I'm especially sorry for this part
        okay = int(np.round(float(okay) - 0.1, 0))
        return okay

    def get_prep_df(self, id):
        prep_df = self.df[self.df['ImageId'] == id]['WKT_Pix']
        prep_df = prep_df.apply(lambda x: x[11:])
        return prep_df

    def get_mask(self, prep_df, im, img_name):
        im_H, im_W, _ = im.shape
        mask = np.zeros((im_H, im_W), np.uint8)
        for _, row in prep_df.iteritems():
            stripped_str = self.split_row(row.strip("(,),' '"))
            start_point = []
            counter = 0
            for elem in stripped_str:
                next_point = (elem.split())
                if counter == 0:
                    start_point = next_point
                if counter > 0:
                    cv2.line(mask, (self.weird_shit(start_point[0]), self.weird_shit(start_point[1])),
                             (self.weird_shit(next_point[0]), self.weird_shit(next_point[1])), (255, 255, 255), 10)
                    start_point = next_point
                counter += 1
        if np.max(mask) == 0:
            print(img_name)
        return mask

    def get_image(self, im_id):
        path = os.path.join(self.path, im_id.rsplit('_i')[0]+'_Roads_Train', self.img_type,
                            self.img_type+'_'+im_id + '.tif')
        im = tifffile.imread(path)
        return im

    def get_weights(self, mask):
        kernel = (3, 3)
        iters = 5
        weights = (self.weight - 1) * cv2.erode(mask, kernel, iters) + 1
        #print(np.average(weights), np.max(weights))
        return weights

    def rotate_bound(self, im, angle):
        (h, w) = im.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        output = cv2.resize(cv2.warpAffine(im, M, (nW, nH)), (h, w), cv2.BORDER_REFLECT_101)
        return output

    def perspective(self, img, shrink_ratio1, shrink_ratio2):
        rows = img.shape[0]
        cols = img.shape[1]

        zero_point = rows - np.round(rows * shrink_ratio1, 0)
        max_point_row = np.round(rows * shrink_ratio1, 0)
        max_point_col = np.round(cols * shrink_ratio2, 0)

        src = np.float32([[zero_point, zero_point], [max_point_row - 1, zero_point], [zero_point, max_point_col + 1],
                          [max_point_row - 1, max_point_col + 1]])
        dst = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])

        perspective_M = cv2.getPerspectiveTransform(src, dst)

        img = cv2.warpPerspective(img, perspective_M, (cols, rows))
        return img


    def random_crop(self, image, mask):
        im_H, im_W, c = image.shape
        slice_size = self.img_size
        i = 0
        while i < 0.01:
            x0 = np.random.randint(0, im_W - slice_size - 1)
            y0 = np.random.randint(0, im_H - slice_size - 1)
            i = np.sum(mask[y0:y0 + slice_size, x0:x0 + slice_size]) / (slice_size * slice_size)
        return image, mask

    def _get_batches_of_transformed_samples(self, index_array):
        imgs = []
        masks = []
        weights = []
        zeros = []
        for elem in index_array:
            img_name = self.dataset[elem]
            im = self.get_image(img_name)
            ms = self.get_mask(self.get_prep_df(img_name), im, img_name)

            im = cv2.resize(im, (self.img_size, self.img_size))
            ms = cv2.resize(ms / 255., (self.img_size, self.img_size))

            if self.is_train:
                rnd_rotate = np.random.randint(4, dtype=int)
                if rnd_rotate > 0:
                    angle = np.random.randint(1, 90)
                    im = self.rotate_bound(im, angle)
                    ms = self.rotate_bound(ms, angle)


                rnd_flip = np.random.randint(2, dtype=int)
                if rnd_flip == 1:
                    rnd_flip = np.random.randint(3, dtype=int)
                    im = cv2.flip(im, rnd_flip - 1)
                    ms = cv2.flip(ms, rnd_flip - 1)


                rnd_perspective = np.random.randint(2, dtype=int)
                if rnd_perspective:
                    shrink_ratio1 = np.random.randint(low=90, high=105, dtype=int) / 100
                    shrink_ratio2 = np.random.randint(low=90, high=105, dtype=int) / 100
                    im = self.perspective(im, shrink_ratio1, shrink_ratio2)
                    ms = self.perspective(ms, shrink_ratio1, shrink_ratio2)

            if self.use_weights:
                w = self.get_weights(ms)
                w = w[..., None]
                z = np.zeros((1024, 1024))
                z = z[..., None]

            ms = np.round(ms)
            ms = ms[..., None]
            imgs.append(im)
            masks.append(ms)


            if self.use_weights: weights.append(w), zeros.append(z)
        imgs = np.array(imgs)
        masks = np.array(masks)
        if self.use_weights:
            weights = np.array(weights)
            zeros = np.array(zeros)
            return [imgs, weights, masks], masks
        else:
            return imgs, masks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)

    def __next__(self):
        with self.lock:
            index_array = next(self.index_generator)[0]
        return self._get_batches_of_transformed_samples(index_array)


