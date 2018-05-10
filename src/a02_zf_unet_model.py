# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np
import sys
from fixed_nets.resnet50_fixed import ResNet50_Fixed_Shapes
from fixed_nets.vgg16_fixed import VGG16_multi_channels
from fixed_nets.inception_resnet_v2 import InceptionResNetV2
# from a00_augmentation_functions import get_model_memory_usage
sys.setrecursionlimit(5000)


# Somehow it needed close values to standard ResNet preprocessing
def preprocess_batch_resnet(batch):
    batch -= 127
    return batch


def preprocess_batch(batch):
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_bce_loss(y_true, y_pred):
    from keras.losses import binary_crossentropy
    return 1 + binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def multi_conv_layer(x, layers, size, dropout, batch_norm):
    from keras import backend as K
    from keras.layers import Conv2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import SpatialDropout2D, Activation

    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = -1

    for i in range(layers):
        x = Conv2D(size, (3, 3), padding='same')(x)
        if batch_norm is True:
            x = BatchNormalization(axis=axis)(x)
        x = Activation('relu')(x)
    if dropout > 0:
        x = SpatialDropout2D(dropout)(x)
    return x


def ZF_Seg_ResNet50_224x224(dropout_val=0.0, batch_norm=True, classes=1):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization
    from keras.layers.core import Activation
    from keras.layers.merge import concatenate
    from keras.applications.vgg16 import VGG16
    # K.set_image_dim_ordering('th')

    if K.image_dim_ordering() == 'th':
        inputs = (3, 224, 224)
        axis = 1
    else:
        inputs = (224, 224, 3)
        axis = 3

    filters = 6

    base_model = ResNet50_Fixed_Shapes(include_top=False, input_shape=inputs, weights='imagenet')
    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    inp = base_model.input

    vgg = VGG16(input_shape=inputs, input_tensor=bn, include_top=False, weights='imagenet')
    vgg_first_conv = vgg.get_layer("block1_conv2").output

    if 0:
        activation_40 = base_model.get_layer("activation_40").output
        activation_22 = base_model.get_layer("activation_22").output
        activation_10 = base_model.get_layer("activation_10").output
        activation_1 = base_model.get_layer("activation_1").output
    else:
        activation_40 = base_model.layers[140].output
        activation_22 = base_model.layers[78].output
        activation_10 = base_model.layers[36].output
        activation_1 = base_model.layers[3].output

    up_14 = concatenate([UpSampling2D(size=(2, 2))(x), activation_40], axis=axis)
    conv_up_14 = multi_conv_layer(up_14, 3, 64 * filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(conv_up_14), activation_22], axis=axis)
    conv_up_28 = multi_conv_layer(up_28, 2, 32 * filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(conv_up_28), activation_10], axis=axis)
    conv_up_56 = multi_conv_layer(up_56, 2, 16 * filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(conv_up_56), activation_1], axis=axis)
    conv_up_112 = multi_conv_layer(up_112, 2, 8 * filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(conv_up_112), vgg_first_conv], axis=axis)
    conv_up_224 = multi_conv_layer(up_224, 2, 8 * filters, 0.2, batch_norm)

    conv_final = Conv2D(classes, (1, 1))(conv_up_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs=inp, outputs=conv_final)
    # print(model.summary())
    return model


def ZF_Seg_ResNet50_224x224_multi_channel(input_ch, dropout_val=0.0, batch_norm=True):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, UpSampling2D
    from keras.layers.core import Activation
    from keras.layers.merge import concatenate
    K.set_image_dim_ordering('th')

    if K.image_dim_ordering() == 'th':
        inputs_3 = (3, 224, 224)
        inputs_4 = (input_ch, 224, 224)
        axis = 1
    else:
        inputs_3 = (224, 224, 3)
        inputs_4 = (224, 224, input_ch)
        axis = 3

    filters = 6

    # Create head for new model
    base_model = ResNet50_Fixed_Shapes(include_top=False, input_shape=inputs_4, weights=None)

    if 1:
        model2 = ResNet50_Fixed_Shapes(include_top=False, input_shape=inputs_3, weights='imagenet')

        # Recalculate weights on first layer
        (weights, bias) = model2.layers[1].get_weights()
        new_weights = np.zeros((weights.shape[0], weights.shape[1], input_ch, weights.shape[3]), dtype=np.float32)
        for i in range(input_ch):
            new_weights[:, :, i, :] = weights[:, :, i % 3, :].copy()
        new_weights = new_weights * 3. / input_ch
        base_model.layers[1].set_weights((new_weights.copy(), bias.copy()))

        # Copy all other weights
        for i in range(2, len(base_model.layers)):
            layer1 = base_model.layers[i]
            layer2 = model2.layers[i]
            layer1.set_weights(layer2.get_weights())

    x = base_model.layers[-2].output
    del base_model.layers[-1:]
    inp = base_model.input

    if 0:
        activation_40 = base_model.get_layer("activation_40").output
        activation_22 = base_model.get_layer("activation_22").output
        activation_10 = base_model.get_layer("activation_10").output
        activation_1 = base_model.get_layer("activation_1").output
    else:
        activation_40 = base_model.layers[140].output
        activation_22 = base_model.layers[78].output
        activation_10 = base_model.layers[36].output
        activation_1 = base_model.layers[3].output

    up_14 = concatenate([UpSampling2D(size=(2, 2))(x), activation_40], axis=axis)
    conv_up_14 = multi_conv_layer(up_14, 3, 64 * filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(conv_up_14), activation_22], axis=axis)
    conv_up_28 = multi_conv_layer(up_28, 2, 32 * filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(conv_up_28), activation_10], axis=axis)
    conv_up_56 = multi_conv_layer(up_56, 2, 16 * filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(conv_up_56), activation_1], axis=axis)
    conv_up_112 = multi_conv_layer(up_112, 2, 8 * filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(conv_up_112), inp], axis=axis)
    conv_up_224 = multi_conv_layer(up_224, 2, 4 * filters, 0.2, batch_norm)

    conv_final = Conv2D(1, (1, 1))(conv_up_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs=inp, outputs=conv_final)
    # print(model.summary())
    return model


def ZF_Seg_VGG16_224x224(dropout_val=0.0, batch_norm=True):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, UpSampling2D
    from keras.layers.core import Activation
    from keras.layers.merge import concatenate
    from keras.applications.vgg16 import VGG16
    # K.set_image_dim_ordering('th')

    # Похоже сходится медленнее когда фильтров много, на 8 было быстрее
    filters = 16

    if K.image_dim_ordering() == 'th':
        inputs = (3, 224, 224)
        axis = 1
    else:
        inputs = (224, 224, 3)
        axis = 3

    base_model = VGG16(include_top=False, input_shape=inputs, weights='imagenet')
    x = base_model.layers[-1].output

    if 0:
        conv_v_14 = base_model.get_layer("block5_conv3").output
        conv_v_28 = base_model.get_layer("block4_conv3").output
        conv_v_56 = base_model.get_layer("block3_conv3").output
        conv_v_112 = base_model.get_layer("block2_conv2").output
        conv_v_224 = base_model.get_layer("block1_conv2").output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv_v_14 = base_model.layers[17].output
        conv_v_28 = base_model.layers[13].output
        conv_v_56 = base_model.layers[9].output
        conv_v_112 = base_model.layers[5].output
        conv_v_224 = base_model.layers[2].output

    conv6 = multi_conv_layer(x, 3, 64 * filters, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv_v_14], axis=axis)
    conv7 = multi_conv_layer(up6, 3, 32 * filters, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv_v_28], axis=axis)
    conv8 = multi_conv_layer(up7, 2, 16 * filters, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv_v_56], axis=axis)
    conv9 = multi_conv_layer(up8, 2, 8 * filters, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv_v_112], axis=axis)
    conv10 = multi_conv_layer(up9, 2, 4 * filters, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv_v_224], axis=axis)
    conv11 = multi_conv_layer(up10, 2, 2 * filters, 0.2, batch_norm)

    conv12 = Conv2D(1, (1, 1))(conv11)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=base_model.input, outputs=conv12)
    return model


def ZF_Seg_VGG16_224x224_directions(dropout_val=0.0, batch_norm=True):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, UpSampling2D
    from keras.layers.core import Activation
    from keras.layers.merge import concatenate
    from keras.applications.vgg16 import VGG16
    # K.set_image_dim_ordering('th')

    # Похоже сходится медленнее когда фильтров много, на 8 было быстрее
    filters = 16

    if K.image_dim_ordering() == 'th':
        inputs = (3, 224, 224)
        axis = 1
    else:
        inputs = (224, 224, 3)
        axis = 3

    base_model = VGG16(include_top=False, input_shape=inputs, weights='imagenet')
    x = base_model.layers[-1].output

    if 0:
        conv_v_14 = base_model.get_layer("block5_conv3").output
        conv_v_28 = base_model.get_layer("block4_conv3").output
        conv_v_56 = base_model.get_layer("block3_conv3").output
        conv_v_112 = base_model.get_layer("block2_conv2").output
        conv_v_224 = base_model.get_layer("block1_conv2").output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv_v_14 = base_model.layers[17].output
        conv_v_28 = base_model.layers[13].output
        conv_v_56 = base_model.layers[9].output
        conv_v_112 = base_model.layers[5].output
        conv_v_224 = base_model.layers[2].output

    conv6 = multi_conv_layer(x, 3, 64 * filters, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv_v_14], axis=axis)
    conv7 = multi_conv_layer(up6, 3, 32 * filters, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv_v_28], axis=axis)
    conv8 = multi_conv_layer(up7, 2, 16 * filters, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv_v_56], axis=axis)
    conv9 = multi_conv_layer(up8, 2, 8 * filters, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv_v_112], axis=axis)
    conv10 = multi_conv_layer(up9, 2, 4 * filters, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv_v_224], axis=axis)
    conv11 = multi_conv_layer(up10, 2, 2 * filters, 0.2, batch_norm)

    conv12 = Conv2D(9, (1, 1))(conv11)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=base_model.input, outputs=conv12)
    return model


def ZF_Seg_VGG16_224x224_multi_channel(input_ch, dropout_val=0.0, batch_norm=True):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, UpSampling2D
    from keras.layers.core import Activation
    from keras.layers.merge import concatenate
    from keras.applications.vgg16 import VGG16
    # K.set_image_dim_ordering('th')

    if K.image_dim_ordering() == 'th':
        inputs_3 = (3, 224, 224)
        inputs_4 = (input_ch, 224, 224)
        axis = 1
    else:
        inputs_3 = (224, 224, 3)
        inputs_4 = (224, 224, input_ch)
        axis = 3

    # Create head for new model
    base_model = VGG16_multi_channels(include_top=False, input_shape=inputs_4)

    # Read ImageNet standard VGG16
    model2 = VGG16(include_top=False, input_shape=inputs_3, weights='imagenet')

    # Recalculate weights on first layer
    (weights, bias) = model2.layers[1].get_weights()
    new_weights = np.zeros((weights.shape[0], weights.shape[1], input_ch, weights.shape[3]), dtype=np.float32)
    for i in range(input_ch):
        new_weights[:, :, i, :] = weights[:, :, i % 3, :].copy()
    new_weights = new_weights * 3. / input_ch
    base_model.layers[1].set_weights((new_weights.copy(), bias.copy()))

    # Copy all other weights
    for i in range(2, len(base_model.layers)):
        layer1 = base_model.layers[i]
        layer2 = model2.layers[i]
        layer1.set_weights(layer2.get_weights())

    x = base_model.layers[-1].output

    if 0:
        conv_v_14 = base_model.get_layer("block5_conv3").output
        conv_v_28 = base_model.get_layer("block4_conv3").output
        conv_v_56 = base_model.get_layer("block3_conv3").output
        conv_v_112 = base_model.get_layer("block2_conv2").output
        conv_v_224 = base_model.get_layer("block1_conv2").output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv_v_14 = base_model.layers[17].output
        conv_v_28 = base_model.layers[13].output
        conv_v_56 = base_model.layers[9].output
        conv_v_112 = base_model.layers[5].output
        conv_v_224 = base_model.layers[2].output

    conv6 = multi_conv_layer(x, 2, 512, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv_v_14], axis=axis)
    conv7 = multi_conv_layer(up6, 2, 256, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv_v_28], axis=axis)
    conv8 = multi_conv_layer(up7, 1, 128, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv_v_56], axis=axis)
    conv9 = multi_conv_layer(up8, 1, 64, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv_v_112], axis=axis)
    conv10 = multi_conv_layer(up9, 1, 32, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv_v_224], axis=axis)
    conv11 = multi_conv_layer(up10, 1, 16, 0.2, batch_norm)

    conv12 = Conv2D(1, (1, 1))(conv11)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=base_model.input, outputs=conv12)

    return model


def ZF_UNET_224(dropout_val=0.05, batch_norm=True):
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Activation
    from keras.layers.merge import concatenate
    inputs = Input((4, 224, 224))

    conv1 = multi_conv_layer(inputs, 2, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = multi_conv_layer(pool1, 2, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = multi_conv_layer(pool2, 2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = multi_conv_layer(pool3, 2, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = multi_conv_layer(pool4, 2, 512, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = multi_conv_layer(pool5, 3, 1024, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=1)
    conv7 = multi_conv_layer(up6, 2, 512, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=1)
    conv8 = multi_conv_layer(up7, 2, 256, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=1)
    conv9 = multi_conv_layer(up8, 2, 128, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=1)
    conv10 = multi_conv_layer(up9, 2, 64, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=1)
    conv11 = multi_conv_layer(up10, 2, 32, 0.2, batch_norm)

    conv12 = Conv2D(1, (1, 1))(conv11)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(inputs=inputs, outputs=conv12)
    return model


"""
Unet with Inception Resnet V2 encoder
"""

def ZF_Seg_Inception_ResNet_v2_288x288():
    from keras.models import Model
    from keras.layers.merge import concatenate
    from keras.layers.convolutional import UpSampling2D
    from keras.layers import Conv2D
    from keras import backend as K

    if K.image_dim_ordering() == 'th':
        inputs = (3, 256, 256)
        axis = 1
    else:
        inputs = (256, 256, 3)
        axis = -1

    filters = 16
    base_model = InceptionResNetV2(include_top=False, input_shape=inputs, weights='imagenet')

    # Freeze encoder Inception_Resnet_v2 layers
    for i in range(len(base_model.layers)):
        base_model.layers[i].trainable = False

    if 0:
        conv1 = base_model.get_layer('activation_3').output
        conv2 = base_model.get_layer('activation_5').output
        conv3 = base_model.get_layer('block35_10_ac').output
        conv4 = base_model.get_layer('block17_20_ac').output
        conv5 = base_model.get_layer('conv_7b_ac').output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv1 = base_model.layers[9].output
        conv2 = base_model.layers[16].output
        conv3 = base_model.layers[260].output
        conv4 = base_model.layers[594].output
        conv5 = base_model.layers[779].output

    conv_inter = multi_conv_layer(conv5, 3, 64 * filters, 0, True)

    up6 = concatenate([UpSampling2D()(conv_inter), conv4], axis=axis)
    conv6 = multi_conv_layer(up6, 2, 32*filters, 0.0, True)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=axis)
    conv7 = multi_conv_layer(up7, 2, 16*filters, 0.0, True)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=axis)
    conv8 = multi_conv_layer(up8, 2, 8*filters, 0.0, True)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=axis)
    conv9 = multi_conv_layer(up9, 2, 4*filters, 0.0, True)

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=axis)
    conv10 = multi_conv_layer(up10, 2, 3*filters, 0.2, True)

    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)

    return model


def ZF_Seg_Inception_ResNet_v2_256x256():
    from keras.models import Model
    from keras.layers.merge import concatenate
    from keras.layers.convolutional import UpSampling2D
    from keras.layers import Conv2D
    from keras import backend as K

    if K.image_dim_ordering() == 'th':
        inputs = (3, 256, 256)
        axis = 1
    else:
        inputs = (256, 256, 3)
        axis = -1

    base_model = InceptionResNetV2(include_top=False, input_shape=inputs, weights='imagenet')

    if 0:
        conv1 = base_model.get_layer('activation_3').output
        conv2 = base_model.get_layer('activation_5').output
        conv3 = base_model.get_layer('block35_10_ac').output
        conv4 = base_model.get_layer('block17_20_ac').output
        conv5 = base_model.get_layer('conv_7b_ac').output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv1 = base_model.layers[9].output
        conv2 = base_model.layers[16].output
        conv3 = base_model.layers[260].output
        conv4 = base_model.layers[594].output
        conv5 = base_model.layers[779].output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=axis)
    conv6 = multi_conv_layer(up6, 2, 256, 0.0, True)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=axis)
    conv7 = multi_conv_layer(up7, 2, 256, 0.0, True)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=axis)
    conv8 = multi_conv_layer(up8, 2, 128, 0.0, True)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=axis)
    conv9 = multi_conv_layer(up9, 2, 64, 0.0, True)

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=axis)
    conv10 = multi_conv_layer(up10, 2, 48, 0.2, True)

    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)

    return model


def ZF_Seg_Inception_ResNet_v2_288x288_multi_channel(input_ch):
    from keras.models import Model
    from keras.layers.merge import concatenate
    from keras.layers.convolutional import UpSampling2D
    from keras.layers import Conv2D
    from keras import backend as K

    if K.image_dim_ordering() == 'th':
        inputs_3 = (3, 288, 288)
        inputs_4 = (input_ch, 288, 288)
        axis = 1
    else:
        inputs_3 = (288, 288, 3)
        inputs_4 = (288, 288, input_ch)
        axis = -1

    # Create head for new model
    base_model = InceptionResNetV2(include_top=False, input_shape=inputs_4, weights=None)

    if 1:
        model2 = InceptionResNetV2(include_top=False, input_shape=inputs_3, weights='imagenet')

        # Recalculate weights on first layer
        (weights, ) = model2.layers[1].get_weights()
        new_weights = np.zeros((weights.shape[0], weights.shape[1], input_ch, weights.shape[3]), dtype=np.float32)
        for i in range(input_ch):
            new_weights[:, :, i, :] = weights[:, :, i % 3, :].copy()
        new_weights = new_weights * 3. / input_ch
        base_model.layers[1].set_weights((new_weights.copy(), ))

        # Copy all other weights
        for i in range(2, len(base_model.layers)):
            layer1 = base_model.layers[i]
            layer2 = model2.layers[i]
            layer1.set_weights(layer2.get_weights())

    if 0:
        conv1 = base_model.get_layer('activation_3').output
        conv2 = base_model.get_layer('activation_5').output
        conv3 = base_model.get_layer('block35_10_ac').output
        conv4 = base_model.get_layer('block17_20_ac').output
        conv5 = base_model.get_layer('conv_7b_ac').output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv1 = base_model.layers[9].output
        conv2 = base_model.layers[16].output
        conv3 = base_model.layers[260].output
        conv4 = base_model.layers[594].output
        conv5 = base_model.layers[779].output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=axis)
    conv6 = multi_conv_layer(up6, 2, 256, 0.0, True)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=axis)
    conv7 = multi_conv_layer(up7, 2, 256, 0.0, True)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=axis)
    conv8 = multi_conv_layer(up8, 2, 128, 0.0, True)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=axis)
    conv9 = multi_conv_layer(up9, 2, 64, 0.0, True)

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=axis)
    conv10 = multi_conv_layer(up10, 2, 48, 0.2, True)

    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)

    return model