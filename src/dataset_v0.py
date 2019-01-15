import numpy as np
from src.norms import Normalize
from tqdm import tqdm
import glob
from src.augs import RandomHueSaturationValue, RandomContrast, RandomBrightness, ShiftScaleRotate, Distort1
from src.utils import get_contours, Integral_image
from copy import deepcopy
import cv2
import os
import random
from keras.preprocessing.image import Iterator
from src.augs_zfturbo import *

if os.name == 'nt':
    GLOBAL_PATH = os.path.join('E:\\', 'Datasets', 'dsbowl2018')
else:
    GLOBAL_PATH = '/home/nosok/datasets/dsBowl2018/'


class BowlDataset(object):
    def __init__(self, path=None, phase='train',
                 train_fraction=0.8,
                 seed=None,
                 normalize='imagenet',
                 im_size=416,
                 dataset_return_ids_only=False):
        self.os_name = os.name
        if path == None:
            self.path = GLOBAL_PATH
        else:
            self.path = path
        self.phase = phase
        if self.phase == 'train':
            self.train_fraction = train_fraction
        else:
            self.train_fraction = 1
        self.all_data = self.get_files_ids()
        if seed is None:
            self.seed = np.random.randint(1, 1000)
            print('Using seed: ', self.seed)
        else:
            self.seed = seed
        self.im_size = im_size
        if dataset_return_ids_only==False:
            self.integral_image = Integral_image(stride=16, kernel=[self.im_size, self.im_size])
            self.dataset_dict, self.train_keys, self.additional_keys = self.create_dict()
        self.test_data = self.all_data

    def get_image(self, idx):
        img_path = os.path.join(self.path, self.phase, idx, 'images', idx+'.png')
        rewrite_tiff = False
        if len(glob.glob(img_path)) == 0:
            img_path = os.path.join(self.path, self.phase, idx, 'images', idx + '.tif')
            rewrite_tiff = True
        img = cv2.imread(img_path, 1)
        if rewrite_tiff:
            print(img_path[:-4]+'.png')
            cv2.imwrite(img_path[:-4]+'.png', img)
        return img

    def get_distance(self, mask):
        mask = np.round(mask)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        dist_transform = cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 5)
        return dist_transform

    def get_weights(self, mask, mask_shapes):
        mask = mask / 255.
        dist_ = self.get_distance(mask.astype(np.uint8))
        dist_ = cv2.normalize(dist_, dst=dist_, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).astype(np.float32)
        # smooth = np.max((mask_shapes[0], mask_shapes[1]))
        pixels = np.sum(mask)
        smooth = np.exp(-pixels / np.max((mask_shapes[0], mask_shapes[1]))).clip(min=0, max=1)
        weights = (1 - dist_) * (1 + np.exp(-pixels / 200)) * mask
        # weights = cv2.normalize(weights, dst=weights, alpha=1, beta=10, norm_type=cv2.NORM_MINMAX).astype(np.float32)
        return weights, dist_

    def get_mask(self, idx):
        mask_path = glob.glob(os.path.join(GLOBAL_PATH, 'train', idx, 'masks', '*.*'))
        mask = cv2.imread(mask_path[0], 0)
        mask_shapes = mask.shape
        boundaries = get_contours(mask)
        # weights, dist = self.get_weights(mask, mask_shapes)
        for m in mask_path[1:]:
            m = cv2.imread(m, 0)
            boundaries += get_contours(m)
            mask += m
            # weights_, dist_ = self.get_weights(m, mask_shapes)
            # weights += weights_
            # dist = np.fmax(dist_, dist)
        mask_ = np.round(mask.astype(np.float32) / 255.)
        boundaries_ = np.round(boundaries.astype(np.float32) / 255.)
        # weights_ = (weights.clip(max=3) + 1.).astype(np.float32)
        weights_ = np.zeros_like(mask_)
        # print(np.min(weights_), np.max(weights_))
        seeds_ = (mask_ - boundaries_).clip(min=0)
        return cv2.merge((mask_, seeds_, weights_)) # FIXME: change dist to seeds

    def get_files_ids(self):
        files_list = []
        for idx in glob.glob(os.path.join(self.path, self.phase, '*')):
            if self.os_name == 'nt':
                files_list.append(idx.rsplit('\\')[-1:][0])
            else:
                files_list.append(idx.rsplit('/')[-1:][0])
        return files_list

    def create_dict(self):
        bowl_dict = {}
        for idx in tqdm(self.all_data):
            im, ms = self.get_image(idx), self.get_mask(idx)
            im, ms, coords = self.integral_image(im, ms)
            bowl_dict[idx] = (im, ms, coords)
        all_keys = list(bowl_dict.keys())
        train_keys = [x for x in all_keys if 'TCGA' not in x and 'gnf' not in x and 'ic100' not in x]
        additional_keys = [x for x in all_keys if 'TCGA' in x or 'gnf' in x or 'ic100' in x]
        return bowl_dict, train_keys, additional_keys

class BowlIterator(Iterator):
    def __init__(self, lock, dataset, datagen=None, predict_seeds=False, is_train=False,
                 batch_size=1, unet_size=512, path=None, seed=None, shuffle=False,
                 normalize='imagenet', n_channels=3, hue_br_contr=False):
        self.bowl_dict = dataset
        self.dataset = list(dataset.keys())
        self.n_channels = n_channels
        self.len = len(self.dataset)
        self.datagen = datagen
        self.predict_seeds = predict_seeds
        self.is_train = is_train
        self.batch_size = batch_size
        self.im_size = unet_size
        self.hue_br_contr = hue_br_contr
        if path is None:
            self.path = GLOBAL_PATH
        else:
            self.path = path
        if seed is None:
            self.seed = np.random.randint(1, 1000)
        else:
            self.seed = seed
        self.lock = lock
        self.shuffle = shuffle
        self.normalize = Normalize(normalize)
        self.RandomHueSaturationValue = RandomHueSaturationValue()
        self.ShiftScaleRotate = ShiftScaleRotate(shift_limit=0.3, scale_limit=0.5, rotate_limit=0)  # shift_limit=0.5265, scale_limit=0.4)
        self.RandomBrightness = RandomBrightness(limit=0.2, prob=0.5)
        self.RandomContrast = RandomContrast(limit=0.2, prob=0.5)
        self.Distort = Distort1(prob=0.2)
        super(BowlIterator, self).__init__(self.len, self.batch_size, shuffle, seed)

    def random_put(self, image, mask):
        shapes = mask.shape
        point1 = np.random.randint(0, self.im_size - shapes[0])
        point2 = np.random.randint(0, self.im_size - shapes[1])
        image_canvas = np.zeros(shape=(self.im_size, self.im_size, self.n_channels), dtype=np.float32)
        mask_canvas = np.zeros(shape=(self.im_size, self.im_size, shapes[2]), dtype=np.float32)
        image_canvas[point1:point1 + shapes[0], point2:point2 + shapes[1]] = image
        mask_canvas[point1:point1 + shapes[0], point2:point2 + shapes[1]] = mask
        return image_canvas, mask_canvas

    def random_crop(self, image, mask, coords):
        points = random.choice(coords)
        image = image[points[0]:points[0]+self.im_size, points[1]:points[1]+self.im_size]
        mask = mask[points[0]:points[0]+self.im_size, points[1]:points[1]+self.im_size]
        return image, mask

    def random_augment_color_for_image(self, img):
        img1 = img.copy()
        separate_channel = True
        if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 0], img[:, :, 2]):
            separate_channel = False
        if np.random.uniform(0, 1) < 0.3:
            img1 = random_brightness_change(img1, 0, 30)
        if np.random.uniform(0, 1) < 0.3:
            img1 = random_dynamic_range_change(img, 0, 20, 0, 20, separate_channel=separate_channel)
        if np.random.uniform(0, 1) < 0.3:
            mean_val = img.mean()
            min_change = -20
            max_change = 20
            if mean_val < 50:
                min_change = 0
            if mean_val > 200:
                max_change = 0
            img1 = random_intensity_change(img1, min_change, max_change, separate_channel=separate_channel)
        if np.random.uniform(0, 1) < 0.2:
            img1 = apply_brightness_renormalization(img1, 1, 5, 8, 12)
        return img1

    def _get_batches_of_transformed_samples(self, index_array):
        imgs = []
        masks = []
        for elem in index_array:
            img_, msk_, coords = self.bowl_dict[self.dataset[elem]]
            img, msk = deepcopy(img_), deepcopy(msk_)

            if self.is_train:
                img = self.random_augment_color_for_image(img.astype(np.uint8))

            if self.normalize is not None:
                img = self.normalize(img)
            img = img.astype(np.float32)

            img, msk = cv2.resize(img, (self.im_size, self.im_size)), cv2.resize(msk, (self.im_size, self.im_size))
            if self.is_train:
                img, msk = self.ShiftScaleRotate(img, msk)
                img, msk = self.Distort(img, msk)

            if self.is_train:
                rnd_flip = np.random.randint(4, dtype=int)
                if rnd_flip > 1:
                    rnd_flip = np.random.randint(3, dtype=int)
                    img = cv2.flip(img, rnd_flip - 1)
                    msk = cv2.flip(msk, rnd_flip - 1)

            img, msk = cv2.resize(img, (self.im_size, self.im_size)), cv2.resize(msk, (self.im_size, self.im_size))

            masks_, seeds_, weights_ = cv2.split(msk)
            if self.predict_seeds:
                masks.append(cv2.merge((np.round(masks_), np.round(seeds_))))
            else:
                masks.append(masks_[..., None])
                weights_ = np.ones((self.im_size, self.im_size), dtype=np.float32)
            imgs.append(img)
        imgs = np.array(imgs)
        masks = np.array(masks)
        return imgs, masks

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)

