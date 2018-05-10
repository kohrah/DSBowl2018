import numpy as np
import os
from src.dataset_v0 import BowlDataset, BowlIterator
from src.predictor_v0 import Predictor
from sklearn.model_selection import KFold
from src.zoo_losses_K import dice_coef_and_binary_loss
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Nadam, SGD
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import threading
from clr import CyclicLR
import src.zoo_losses_K as zoo_losses_K
import src.zoo_unet_K as zoo_unet_K
from src.a02_zf_unet_model import ZF_Seg_ResNet50_224x224

if os.name == 'nt':
    GLOBAL_PATH = 'E:\Datasets\\dsbowl2018\\'
else:
    GLOBAL_PATH = '/home/nosok/datasets/dsBowl2018/'

class KFold_cbk(keras.callbacks.Callback):
    def __init__(self, model, fold_n):
        self.fold_n = fold_n
        self.model_to_save = model
        self.best_loss = 1000.
    def on_epoch_end(self, epoch, logs):
        if np.round(logs.get('val_loss'), 4) < self.best_loss:
            self.best_loss = np.round(logs.get('val_loss'), 4)
            print('Saving model to <model_at_fold_BN_%d.h5>' % self.fold_n)
            self.model_to_save.save('model_at_fold_BN_%d.h5' % self.fold_n)

class Trainer(object):
    def __init__(self, model, optimizer, loss, classes=1, seed=1, input_size=416, n_channels=3,
                 batch_size=16, batch_norm=False, dropout=0.2,
                 n_folds=3, exclude_folds=[],
                 lr_schedule=None,
                 train_fraction=0.8,
                 norm_function='imagenet', predict_seeds=True,
                 path=None,
                 use_multiprocessing=False,
                 mode='train',
                 weights_sub_path='weights'):
        self.seed = seed
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.classes = classes
        self.input_size = input_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.n_folds = n_folds
        self.exclude_folds = exclude_folds
        self.train_fraction = train_fraction
        self.norm_function = norm_function
        self.predict_seeds = predict_seeds
        self.mode = mode
        self.use_multiprocessing = use_multiprocessing
        self.dataset, self.train_keys, self.additional_keys = self.get_dataset()
        self.weights_sub_path = weights_sub_path

        if lr_schedule is None:
            self.lr_schedule = [[200, 0.0001, '1']]
        else:
            self.lr_schedule = lr_schedule
        if path is None:
            self.path = GLOBAL_PATH
        else:
            self.path = path

    def get_dataset(self):
        dataset = BowlDataset(train_fraction=self.train_fraction, seed=self.seed)
        train_data, train_ids, additional_ids = dataset.dataset_dict, dataset.train_keys, dataset.additional_keys
        return train_data, train_ids, additional_ids

    def ids_to_keys(self, list_, additional=[]):
        keys = [self.train_keys[i] for i in list_]
        keys = keys + additional
        data = {k: self.dataset[k] for k in keys}
        return data

    def get_generator(self, is_train, dataset, hue_br_contr=False):
        generator = BowlIterator(lock=threading.Lock(), batch_size=self.batch_size, dataset=dataset,
                                 is_train=is_train, predict_seeds=self.predict_seeds, normalize=self.norm_function,
                                 datagen=ImageDataGenerator(),
                                 shuffle=True, hue_br_contr=hue_br_contr,
                                 unet_size=self.input_size, n_channels=self.n_channels)
        return generator

    def train(self):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold = 0
        for train_keys_, val_keys_ in kf.split(self.train_keys):
            if fold in self.exclude_folds:
                print('!!! Skipped fold ' + str(fold))
                fold += 1
                pass
            else:
                # print(train_keys_, self.additional_keys)
                train_data, val_data = self.ids_to_keys(train_keys_, self.additional_keys), self.ids_to_keys(val_keys_)

                train_gen = self.get_generator(is_train=True, dataset=train_data)
                val_gen = self.get_generator(is_train=False, dataset=val_data)

                model_ = zoo_unet_K.get_unet_512_spartial
                # model = model_(input_shape=(self.input_size, self.input_size, self.n_channels), classes=self.classes,
                #                batch_norm=self.batch_norm, n_filters=16, dropout_val=self.dropout)
                model = model_(UNET_INPUT=self.input_size, classes=self.classes,
                               dropout_val=self.dropout, batch_norm=self.batch_norm)
                callback = KFold_cbk(model, fold)
                fold_checkpoint_name = 'model_at_fold_BN_%d.h5' % fold
                for epochs, lr, iter_n in self.lr_schedule:
                    if iter_n == '1' or (iter_n != check_id):
                        with tf.device("/cpu:0"):
                            print('--------\nNew init, fold %d, starting iteration %s' % (fold, iter_n))
                            if iter_n != '1':
                                print('loading weights...')
                                model.load_weights(fold_checkpoint_name)
                                print('Done!')
                            if iter_n not in ['1', '2']:
                                print('VGG layers are trainable now')
                                for l in model.layers:
                                    l.trainable = True
                            # if iter_n not in['1', '2', '3', '4']:
                            #     print('HueSat, Contrast, Brightness are enabled ')
                            #     train_gen = self.get_generator(is_train=True, dataset=train_data, hue_br_contr=True)
                        parallel_model = multi_gpu_model(model, gpus=2)

                    clr = CyclicLR(base_lr=lr * 0.8, max_lr=lr * 2, step_size=120.)

                    parallel_model.compile(optimizer=self.optimizer(lr=lr, clipnorm=1.),
                                           loss=self.loss, metrics=[zoo_losses_K.dice_coef])

                    parallel_model.fit_generator(train_gen,
                                                 steps_per_epoch=np.ceil(len(train_keys_) / self.batch_size),
                                                 epochs=epochs,
                                                 validation_data=val_gen,
                                                 validation_steps=np.ceil(len(val_keys_) / self.batch_size),
                                                 verbose=2,
                                                 callbacks=[callback], # FIXME: turn clr back
                                                 max_queue_size=30,
                                                 use_multiprocessing=self.use_multiprocessing,
                                                 workers=4)
                    check_id = iter_n
                fold += 1
            print('Finished!')
#
# # train
# import zoo_losses_K
# import zoo_unet_K
#
# phase = 'train'
# batch_size = 24
# classes = 2
# batch_norm = True
# dropout = 0.0
# unet_input = 224
# norm_type = 'mean_std'
# predict_seeds = False
# if predict_seeds:
#     classes = 2
# n_folds = 3
# exclude_folds = [1, 2]
# lr_sched = [[10, 0.0003, '1'], [10, 0.0002, '2'], [15, 0.00015, '3'],
#             [90, 0.0001, '4']]#, [20, 0.0003, '5']]
# # lr_sched = [[1, 0.0001, '1'], [1, 0.0005, '2'], [1, 0.0002, '3'],
# #             [1, 0.0001, '4'], [1, 0.00007, '5']]
# use_model = zoo_unet_K.get_VGGunet_bnd_v3
# loss = zoo_losses_K.dice_coef_weighted_one_class #dice_coef_and_binary_loss
# metrics = zoo_losses_K.dice_coef
#
# seed = 66 #980
#
# if os.name == 'nt':
#     GLOBAL_PATH = 'E:\Datasets\\dsbowl2018\\'
# else:
#     GLOBAL_PATH = '/home/nosok/datasets/dsBowl2018/'
#
# if os.name != 'nt':
#     use_multiprocessing = True
# else:
#     use_multiprocessing = False
#
# if phase == 'train':
#     trainer = Trainer(model=use_model, optimizer=Nadam,
#                       batch_size=batch_size, input_size=unet_input, norm_function=norm_type,
#                       loss=loss, classes=classes,
#                       predict_seeds=predict_seeds,
#                       lr_schedule=lr_sched,
#                       n_folds=n_folds, exclude_folds=exclude_folds,
#                       seed=seed,
#                       use_multiprocessing=use_multiprocessing,
#                       batch_norm=batch_norm,
#                       dropout=dropout)
#     trainer.train()
#
# if phase == 'test':
#     Test_dataset = BowlDataset(train_fraction=1, seed=1, phase='test', dataset_return_ids_only=True)
#     test_data = Test_dataset.test_data
#     predictor = Predictor(phase='test', dataset=test_data, model=use_model, model_input_size=unet_input,
#                           path=GLOBAL_PATH,
#                           predict_on_crops=False,
#                           n_classes=classes, normalize=norm_type,
#                           n_folds=n_folds, exclude_folds=exclude_folds,
#                           nb_tta_times=2,
#                           mask_threshold=0.5, seed_threshold=0.6, seed_min_size=4,
#                           debug=True,
#                           tf_device="/gpu:0",
#                           dataset_return_ids_only=True)
#     predictor.predict()
#
