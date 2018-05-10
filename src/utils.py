import numpy as np
import cv2
import os, glob
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from keras import callbacks


if os.name == 'nt':
    GLOBAL_PATH = os.path.join('E:\\', 'Datasets', 'dsbowl2018')
else:
    GLOBAL_PATH = '/media/nosok/2AC016BAC0168C67/Datasets/dsBowl2018/'

class Callback(callbacks.Callback):
    def __init__(self, model, iter_n):
        self.iter_n = iter_n
        self.model_to_save = model
        self.best_loss = 1000.

    def on_epoch_end(self, epoch, logs):
        # print(logs.get('val_loss'))
        if np.round(logs.get('val_loss'), 4) < self.best_loss:
            self.best_loss = np.round(logs.get('val_loss'), 4)
            print('Saving model to <model_at_iter_%s.h5>' % self.iter_n)
            self.model_to_save.save('model_at_iter_%s.h5' % self.iter_n)

class Integral_image(object):
    def __init__(self, stride, kernel=[256, 256], threshold=0.05):
        self.stride = stride
        self.kernel = kernel
        self.threshold = threshold

    def prep_image(self, image, mask, im_w, im_h):
        res_x, res_y = (im_w - self.kernel[0]) % self.stride, (im_h - self.kernel[1]) % self.stride
        im_w_1, im_h_1 = im_w, im_h
        if res_x != 0:
            im_w_1_ = ((im_w - self.kernel[0]) // self.stride + 1) * self.stride + self.kernel[0]
            if (im_w_1_ - im_w) < self.kernel[0] / 2:
                im_w_1 = im_w_1_
        if res_y != 0:
            im_h_1_ = ((im_h - self.kernel[1]) // self.stride + 1) * self.stride + self.kernel[1]
            if (im_h_1_ - im_h) < self.kernel[1] / 2:
                im_h_1 = im_h_1_
        if res_x != 0 or res_y != 0:
            image = cv2.resize(image, (im_w_1, im_h_1))
            mask = cv2.resize(mask, (im_w_1, im_h_1))
        return image, mask, im_w_1, im_h_1

    def __call__(self, image, mask):
        coords_list = []
        im_h, im_w, im_c = mask.shape[0], mask.shape[1], mask.shape[2]
        image, mask, im_w_1, im_h_1 = self.prep_image(image, mask, im_w, im_h)
        area_sq = self.kernel[0] * self.kernel[1]
        x, y = 0, 0
        while y+self.kernel[1] <= im_h_1:
            while x+self.kernel[0] <= im_w_1:
                # [y0:y0+h, x0:x0+w]
                sum_ = np.sum(mask[y:y+self.kernel[1], x:x+self.kernel[0]])
                if sum_ / area_sq > self.threshold:#*im_c
                    coords_list.append([int(np.round(y)), int(np.round(x))])
                x += self.stride
            y += self.stride
            x = 0
        return image, mask, coords_list

class Sliding_window(object):
    def __init__(self, kernel, overlap):
        self.kernel = kernel
        self.overlap = overlap

    def get_new_shape(self, residual, max_point, step_size):
        if residual > 0.5:
            max_point += 1
        max_point = max(1, max_point)
        new_shape = max_point * (step_size - self.overlap)
        return new_shape, max_point

    def __call__(self, im_rows, im_cols):
        max_R, res_R = (im_rows - self.overlap) // (self.kernel[0] - self.overlap), \
                       im_rows % (self.kernel[0] - self.overlap) / self.kernel[0] #im_rows - overlap //2
        max_C, res_C = (im_cols - self.overlap) // (self.kernel[1] - self.overlap), \
                       im_cols % (self.kernel[1] - self.overlap) / self.kernel[1]
        new_R_shape, R_steps = self.get_new_shape(res_R, max_R, self.kernel[0])
        new_C_shape, C_steps = self.get_new_shape(res_C, max_C, self.kernel[1])
        return [new_R_shape, new_C_shape, R_steps, C_steps]

def strider(x):
    pass


def get_image(idx):
    img_path = os.path.join(GLOBAL_PATH, 'train', idx, 'images', idx+'.png')
    img = cv2.imread(img_path, 1)
    return img

def get_mask(idx):
    mask_path = glob.glob(os.path.join(GLOBAL_PATH, 'train', idx, 'masks', '*.*'))
    msk = cv2.imread(mask_path[0], 0)
    mask_dilation_erosion = cv2.erode(cv2.dilate(msk, (3, 3), iterations=1), (3, 3), iterations=1)
    boundaries = ((msk - mask_dilation_erosion) > 0).astype(np.uint8)
    for m in mask_path[1:]:
        m = cv2.imread(m, 0)
        boundaries += ((m - cv2.erode(cv2.dilate(m, (3, 3), iterations=3), (3, 3), iterations=3)) > 0).astype(np.uint8)
        msk += m
    return msk, boundaries

def get_contours(mask):
    mask_contour = np.zeros_like(mask).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(mask_contour, contours, -1, (255, 255, 255), 2)

def get_center(mask):
    mask_center = np.zeros_like(mask).astype(np.uint8)
    y, x = ndi.measurements.center_of_mass(mask)
    cv2.circle(mask_center, (int(x), int(y)), 4, (255, 255, 255), -1)
    return mask_center

def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

import skimage.morphology as morph
from postprocess import relabel
def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)


def watershed_ndi(image, distance=None, markers=None, min_size=10):
    image = image.astype(np.uint8)
    if distance is None:
        distance = ndi.distance_transform_edt(image)
    else:
        distance = distance
    if markers is None:
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((2, 2)), labels=image)
        markers = ndi.label(local_maxi)[0]
    else:
        # markers = cv2.erode(markers, (3,3), iterations=1)
        markers = ndi.label(np.round(markers).astype(bool))[0]
    markers = drop_small(markers, min_size=min_size)
    labels = watershed(-distance, markers, mask=image)
    return labels

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks

import pandas as pd
def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def create_submission(path, test_data): #meta, predictions
    image_ids, encodings = [], []
    # for image_id, prediction in zip(meta['ImageId'].values, predictions):
    for image_id, prediction in enumerate(test_data):
        for mask in decompose(prediction):
            image_ids.append(image_id)
            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask > 128.))
            encodings.append(rle_encoded)

    submission = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encodings}).astype(str)
    submission = submission[submission['EncodedPixels']!='nan']
    submission_filepath = os.path.join(path, 'submission.csv')
    submission.to_csv(submission_filepath, index=None)

