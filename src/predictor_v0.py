import numpy as np
import pandas as pd
import cv2
import os
import src.dataset_v0 as dataset
from src.dataset_v0 import Normalize, BowlDataset
import src.zoo_losses_K as zoo_losses_K
import tensorflow as tf
from keras.optimizers import Adam
from tqdm import tqdm
from src.utils import get_mask, watershed_ndi, get_contours, Callback, Sliding_window
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

# some params

phase = 'test'
n_folds = 3
show_plot = True
batch_size = 16
unet_input = 224
classes = 2
normalize = 'imagenet'


if os.name == 'nt':
    GLOBAL_PATH = 'E:\Datasets\\dsbowl2018\\'
else:
    GLOBAL_PATH = '/home/nosok/datasets/dsBowl2018/'

# ====================================================================================== #
# BODY
Dataset = dataset.BowlDataset(train_fraction=1, seed=1)
Test_dataset = dataset.BowlDataset(train_fraction=1, seed=1, phase='test')
test_data = Test_dataset.test_data
print(len(test_data))

class Predictor(object):
    def __init__(self, phase, dataset, model, model_input_size,
                 path, weights_sub_path='weights', images_sub_path='images',
                 save_csv_to='csv', save_masks_to='test_saves',
                 predict_on_crops=True,
                 n_classes=2, normalize='imagenet',
                 n_folds=5, exclude_folds=None,
                 nb_tta_times=4,
                 crop_step_size=16, border_size=24,
                 mask_threshold=0.5, seed_threshold=0.7, seed_min_size=5,
                 debug=False,
                 tf_device="/gpu:0",
                 dataset_return_ids_only=True):

        self.phase = phase
        self.dataset_return_ids_only = dataset_return_ids_only
        self.dataset = dataset
        self.model = model
        self.input_size = model_input_size
        if path is None:
            self.path = GLOBAL_PATH
        else:
            self.path = path
        self.weights_path = os.path.join(self.path, weights_sub_path)
        self.save_csv_to = os.path.join(self.path, save_csv_to)
        self.save_masks_to = os.path.join(self.path, save_masks_to)
        self.images_sub_path = images_sub_path
        self.predict_on_crops = predict_on_crops
        if self.predict_on_crops:
            self.crop_step_size = crop_step_size
            self.border = border_size
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.exclude_folds = exclude_folds
        self.nb_tta_time = nb_tta_times
        self.mask_threshold = mask_threshold
        self.seed_threshold = seed_threshold
        self.seed_min_size = seed_min_size
        self.debug = debug
        self.normalize = Normalize(normalize)
        self.tf_device = tf_device
        self.dataset_return_ids_only = dataset_return_ids_only

    def clip(self, img, maxval, dtype=np.float32):
        return np.clip(img, 0, maxval).astype(dtype)

    def get_mirror_image_by_index(self, image, index):
        if index < 4:
            image = np.rot90(image, k=index)
        else:
            if len(image.shape) == 3:
                image = image[::-1, :, :]
            else:
                image = image[::-1, :]
            image = np.rot90(image, k=index - 4)
        return image

    def get_mirror_image_by_index_backward(self, image, index):
        if index < 4:
            image = np.rot90(image, k=-index)
        else:
            image = np.rot90(image, k=-(index - 4))
            if len(image.shape) == 3:
                image = image[::-1, :, :]
            else:
                image = image[::-1, :]
        return image

    def get_mask_from_model_v1(self, image):
        box_size = self.input_size
        initial_shape = image.shape
        initial_rows, initial_cols = image.shape[0], image.shape[1]
        if self.debug:
            print('Initial image shape: {}'.format(image.shape))

        if image.shape[0] < box_size or image.shape[1] < box_size:
            new_image = np.zeros((max(image.shape[0], box_size), max(image.shape[1], box_size), image.shape[2]))
            new_image[0:image.shape[0], 0:image.shape[1], :] = image
            image = new_image
            if self.debug:
                print('Rescale image... New shape: {}'.format(image.shape))

        if self.n_classes > 1:
            final_mask = np.zeros((initial_rows, initial_cols, self.n_classes), dtype=np.float32)
        else:
            final_mask = np.zeros(image.shape[:2], dtype=np.float32)

        count = np.zeros(image.shape[:2], dtype=np.float32)
        image_list = []
        params = []

        # 224 cases
        if 1:
            size_of_subimg = self.input_size
            # step = 14
            step = self.crop_step_size  #size_of_subimg // 8  # 28
            for j in range(0, image.shape[0], step):
                for k in range(0, image.shape[1], step):
                    start_0 = j
                    start_1 = k
                    end_0 = start_0 + size_of_subimg
                    end_1 = start_1 + size_of_subimg
                    if end_0 > image.shape[0]:
                        start_0 = image.shape[0] - size_of_subimg
                        end_0 = image.shape[0]
                    if end_1 > image.shape[1]:
                        start_1 = image.shape[1] - size_of_subimg
                        end_1 = image.shape[1]

                    image_part = image[start_0:end_0, start_1:end_1].copy()

                    for i in range(self.nb_tta_time):
                        im = self.get_mirror_image_by_index(image_part.copy(), i)
                        # im = cv2.resize(im, (box_size, box_size), cv2.INTER_LANCZOS4)
                        image_list.append(im)
                        params.append((start_0, start_1, size_of_subimg, i))

                    if k + size_of_subimg >= image.shape[1]:
                        break
                if j + size_of_subimg >= image.shape[0]:
                    break
        if self.debug:
            print('Masks to calc: {}'.format(len(image_list)))
        image_list = np.array(image_list, dtype=np.float32)

        mask_list = self.model.predict(image_list, batch_size=48)

        if self.debug:
            print('Number of masks:', mask_list.shape)

        # border = 20
        for i in range(mask_list.shape[0]):
            if K.image_dim_ordering() == 'th':
                mask = mask_list[i, self.n_classes, :, :].copy()
            else:
                mask = mask_list[i, :, :, :].copy()
            if mask_list[i].shape[0] != params[i][2]:
                mask = cv2.resize(mask, (params[i][2], params[i][2]), cv2.INTER_LANCZOS4)
            mask = self.get_mirror_image_by_index_backward(mask, params[i][3])
            # show_resized_image(255*mask)

            # Find location of mask. Cut only central part for non border part
            if params[i][0] < self.border:
                start_0 = params[i][0]
                mask_start_0 = 0
            else:
                start_0 = params[i][0] + self.border
                mask_start_0 = self.border

            if params[i][0] + params[i][2] >= final_mask.shape[0] - self.border:
                end_0 = params[i][0] + params[i][2]
                mask_end_0 = mask.shape[0]
            else:
                end_0 = params[i][0] + params[i][2] - self.border
                mask_end_0 = mask.shape[0] - self.border

            if params[i][1] < self.border:
                start_1 = params[i][1]
                mask_start_1 = 0
            else:
                start_1 = params[i][1] + self.border
                mask_start_1 = self.border

            if params[i][1] + params[i][2] >= final_mask.shape[1] - self.border:
                end_1 = params[i][1] + params[i][2]
                mask_end_1 = mask.shape[1]
            else:
                end_1 = params[i][1] + params[i][2] - self.border
                mask_end_1 = mask.shape[1] - self.border

            tmp = mask[mask_start_0:mask_end_0, mask_start_1:mask_end_1]
            tmp, _ = cv2.split(tmp)
            # print('sss  ', tmp.shape, final_mask.shape)
            final_mask[start_0:end_0, start_1:end_1] += \
                mask[mask_start_0:mask_end_0, mask_start_1:mask_end_1]
            count[start_0:end_0, start_1:end_1] += 1

        if count.min() == 0:
            print('Some uncovered parts of image!')
        if self.debug:
            print(final_mask.shape)
            print(np.max(final_mask), np.min(final_mask))

        # FIXME correct the multiclass part
        # predicted, pred_seeds = cv2.split(final_mask) / count
        final_mask /= np.stack((count, count), 2)
        if initial_shape[:2] != final_mask.shape[:2]:
            final_mask = final_mask[0:initial_shape[0], 0:initial_shape[1]]
            print('Return shape back: {}'.format(final_mask.shape))

        return final_mask

    def predict(self):
        test_list = []
        nucleus = 0
        test_dict = dict({(k, 0) for k in self.dataset})
        for fold in range(self.n_folds):
            if fold in self.exclude_folds:
                print('Excluded fold: ', fold)
                fold += 1
                pass
            else:
                fold_weights = os.path.join(self.weights_path, 'model_at_fold_BN_%d.h5' % fold)
                with tf.device(self.tf_device):
                    self.model = load_model(fold_weights)
                self.model.compile(optimizer=Adam(lr=0.0001), loss=zoo_losses_K.dice_coef)
                for image_id in tqdm(test_data):
                    if self.debug:
                        print('Predicting %s' % image_id)

                    path_to_im = os.path.join(self.path, self.phase, image_id,
                                              self.images_sub_path, '%s.png' % image_id)

                    original_image = cv2.imread(path_to_im, 1)
                    original_shape = original_image.shape

                    if normalize is not None:
                        transformed_image = self.normalize(original_image)
                    else:
                        transformed_image = original_image

                    if transformed_image.shape[0] < unet_input:
                        transformed_image = cv2.resize(transformed_image, (unet_input, transformed_image.shape[1]))
                    if transformed_image.shape[1] < unet_input:
                        transformed_image = cv2.resize(transformed_image, (transformed_image.shape[0], unet_input))

                    if self.predict_on_crops:
                        # print(transformed_image.shape)
                        predicted_, pred_seeds_ = \
                            cv2.split(self.get_mask_from_model_v1(transformed_image))
                    else:
                        transformed_image_ = cv2.resize(transformed_image, (unet_input, unet_input))
                        if self.n_classes == 1:
                            predicted_ = \
                                cv2.resize(
                                        np.squeeze(self.model.predict(transformed_image_[None, ...]), axis=0)
                                        , (original_shape[1], original_shape[0]))
                        else:
                            predicted_, pred_seeds_ = \
                                cv2.split(
                                    cv2.resize(
                                        np.squeeze(self.model.predict(transformed_image_[None, ...]), axis=0)
                                        , (original_shape[1], original_shape[0])))

                    predicted = \
                        cv2.resize(predicted_, (original_shape[1], original_shape[0])) / (self.n_folds - len(self.exclude_folds))
                    if self.n_classes > 1:
                        pred_seeds =\
                            cv2.resize(pred_seeds_, (original_shape[1], original_shape[0])) / (self.n_folds - len(self.exclude_folds))

                    if self.debug:
                        print('Original image shape: ', original_image.shape)
                        print('Mask params: ', np.min(predicted), np.max(predicted), n_folds, len(self.exclude_folds))
                        print('Seed params: ', np.min(pred_seeds), np.max(pred_seeds))
                    if self.n_classes == 1:
                        test_dict[image_id] += predicted
                    else:
                        test_dict[image_id] += cv2.merge((predicted, pred_seeds))

                test_list_ = []
                for key, value in test_dict.items():
                    test_list_.append([key, value])

        for image_id, merged_mask in test_list_:
            if self.n_classes > 1:
                predicted, pred_seeds = cv2.split(merged_mask)
                watershedded = watershed_ndi(predicted > self.mask_threshold,
                                         markers=(pred_seeds > self.seed_threshold),
                                         min_size=self.seed_min_size)
            if self.debug:
                # print(np.max(watershedded))
                path_to_im = os.path.join(GLOBAL_PATH, phase, image_id, 'images', '%s.png' % image_id)
                original_image = cv2.imread(path_to_im, 1)
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 8), sharex=True, sharey=True)
                ax = axes.ravel()
                ax[0].imshow(original_image, interpolation='nearest', alpha=.7)
                ax[0].set_title("Original image")
                if self.n_classes > 1:
                    ax[1].imshow(watershedded, cmap=plt.cm.nipy_spectral, interpolation='nearest', alpha=.7)
                    ax[1].set_title("Segmented crop")
                # ax[2].imshow(watershedded_, cmap=plt.cm.nipy_spectral, interpolation='nearest', alpha=.7)
                # ax[2].set_title("Segmented resize")
                ax[2].imshow((predicted > 0.5), interpolation='nearest', alpha=.7)
                ax[2].set_title("predicted")
                if self.n_classes > 1:
                    ax[3].imshow((pred_seeds > 0.5), interpolation='nearest', alpha=.7)
                    ax[3].set_title("Seeds")
                fig.tight_layout()
                plt.show()

            if self.n_classes > 1:
                nucleus += np.max(watershedded)
            else:
                watershedded = predicted
            cv2.imwrite(os.path.join(self.path, self.save_masks_to, image_id + '.png'), np.round(predicted)*255)
            test_list.append([image_id, watershedded])

        print('nb_nucleus: ', nucleus)

        from RLE import decompose, run_length_encoding
        image_ids = []
        encodings = []
        for im_id, prediction in test_list:
            for mask in decompose(prediction):
                image_ids.append(im_id)
                encodings.append(' '.join(str(rle) for rle in run_length_encoding(mask > 128.)))
        submission = pd.DataFrame()
        submission['ImageId'] = image_ids
        submission['EncodedPixels'] = encodings
        fname = os.path.join(self.path, self.save_csv_to, 'sub-dsbowl.csv')
        submission.to_csv(fname, index=False)

        print('\nPredicited!')

