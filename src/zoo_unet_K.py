from keras.layers import add, concatenate, Conv2D, MaxPooling2D, Dropout, Conv1D, Reshape
from keras.layers import Input, Convolution2D, UpSampling2D, Conv2DTranspose, AtrousConv2D, \
                         BatchNormalization, Activation, SpatialDropout2D, Cropping2D
from keras.models import Model
from keras.applications import ResNet50, vgg19, vgg16
from keras.layers import LeakyReLU, Lambda, Dense

# ========================= Building blocks =========================
def double_conv_layer(x, size, dropout, batch_norm):
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=-1)(conv)
    conv = Activation('elu')(conv)
    conv = Convolution2D(size, (3, 3), padding='same')(conv)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=-1)(conv)
    conv = Activation('elu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv

def double_conv_layer_v2(x, size, dropout, batch_norm):
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=-1)(conv)
    conv = Activation('elu')(conv)
    conv = Convolution2D(size // 2, (3, 3), padding='same')(conv)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=-1)(conv)
    conv = Activation('elu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv

def double_atrous_conv_layer(x, size, dropout, batch_norm):
    conv = Conv2D(size, (3, 3), padding='same', dilation_rate=8)(x)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=-1)(conv)
    conv = Activation('elu')(conv)
    conv = Convolution2D(size, (3, 3), padding='same', dilation_rate=8)(conv)
    if batch_norm == True:
        conv = BatchNormalization(mode=0, axis=-1)(conv)
    conv = Activation('elu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

# ===================================================================


def get_unet(input_shape=(512, 512, 3), dropout_val=0.1, batch_norm=False):
    inputs = Input(input_shape)
    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = double_conv_layer(up10, 32, 0, batch_norm)

    conv13 = Conv2D(1, (1, 1), activation='sigmoid', name='output')(conv11)

    model = Model(input=inputs, output=conv13)
    return model

def get_unet_512_spartial(UNET_INPUT, dropout_val=0.25, spartial_dropout=0, cropping=(0,0), batch_norm=False, classes=1):
    inputs = Input((UNET_INPUT, UNET_INPUT, 3))
    bn = BatchNormalization(mode=0, axis=-1)(inputs)
    conv1 = double_conv_layer(bn, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = double_conv_layer(pool5, 1024, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
    conv7 = double_conv_layer(up6, 512, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = double_conv_layer(up10, 32, 0, batch_norm)

    if cropping > (0, 0):
        conv11 = Cropping2D(cropping=(cropping, cropping))(conv11)

    if spartial_dropout > 0:
        conv11 = SpatialDropout2D(rate=.5)(conv11)

    conv12 = Conv2D(classes, (1, 1), activation='sigmoid')(conv11)

    model = Model(input=inputs, output=conv12)

    return model



# ========================= Multi-input UNET =========================
def multi_unet(UNET_INPUT, dropout_val=0.2, batch_norm=False, activation='relu'):
    main_input = Input(shape=(UNET_INPUT, UNET_INPUT, 3), name='main_input')
    aux_input = Input(shape=(UNET_INPUT//4, UNET_INPUT//4, 8), name='aux_input')

    conv1 = double_conv_layer(main_input, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm) #256
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm) #128

    aux_conv1 = Conv2D(128, (3, 3), padding='same')(aux_input)
    aux_conv1 = BatchNormalization(mode=0, axis=-1)(aux_conv1)
    aux_conv1 = Activation('elu')(aux_conv1)
    aux_conv1 = Conv2D(128, (3, 3), padding='same', name='checkpoint1')(aux_conv1)
    aux_conv1 = BatchNormalization(mode=0, axis=-1)(aux_conv1)
    aux_conv1 = Activation('elu')(aux_conv1)
    aux_conv1 = Dropout(dropout_val, name='checkpoint2')(aux_conv1)
    #aux_conv1 = Reshape((128, 128, 1))(aux_conv1)

    branch_concat = concatenate([conv3, aux_conv1], axis=-1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(branch_concat)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = double_conv_layer(up10, 32, 0, batch_norm)

    #crop1 = Cropping2D(((32 ,32), (32, 32)))(conv11)

    spartial1 = SpatialDropout2D(rate=.25)(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(spartial1)

    model = Model(input=[main_input, aux_input], output=conv12)
    return model


# ========================= Unet w. Resnet encoder =========================
def get_unet_resnet(input_shape, activation='relu'):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)
    resnet_base.summary()
    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 128, "conv7_1")
    conv7 = conv_block_simple(conv7, 128, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 64, "conv8_1")
    conv8 = conv_block_simple(conv8, 64, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 32, "conv9_1")
    conv9 = conv_block_simple(conv9, 32, "conv9_2")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    conv10 = Cropping2D(cropping=((32, 32), (32, 32)))(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model

# def get_unet_w(input_shape=(512, 512, 3), weights_shape=(512, 512, 1), dropout_val=0.2, batch_norm=False):
#     original_input = Input(input_shape, name='input')
#     mask_weights = Input(weights_shape)
#     true_masks = Input(weights_shape)
#     conv1 = double_conv_layer(original_input, 32, dropout_val, batch_norm)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
#
#     up7 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
#     conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)
#
#     up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
#     conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)
#
#     up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
#     conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)
#
#     up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
#     conv11 = double_conv_layer(up10, 32, 0, batch_norm)
#
#     #spartial1 = SpatialDropout2D(rate=.3)(conv11)
#
#     y_pred = Conv2D(1, (1, 1), activation='sigmoid', name='y_pred')(conv11)
#
#     loss = Lambda(weighted_binary_loss,  trainable=False)([y_pred, mask_weights, true_masks])
#
#     model = Model(inputs=[original_input, mask_weights, true_masks], outputs=loss)
#
#     return model
#
# def get_unet_w_pred(input_shape=(512, 512, 3), weights_shape=(512, 512, 1), dropout_val=0.25, batch_norm=False):
#     original_input = Input(input_shape, name='input')
#     mask_weights = Input(weights_shape)
#     true_masks = Input(weights_shape)
#     conv1 = double_conv_layer(original_input, 32, dropout_val, batch_norm)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
#
#     up7 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
#     conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)
#
#     up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
#     conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)
#
#     up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
#     conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)
#
#     up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
#     conv11 = double_conv_layer(up10, 32, 0, batch_norm)
#
#     #spartial1 = SpatialDropout2D(rate=.3)(conv11)
#
#     y_pred = Conv2D(1, (1, 1), activation='sigmoid', name='y_pred')(conv11)
#
#     loss = Lambda(weighted_binary_loss, trainable=False)([y_pred, mask_weights, true_masks])
#
#     model = Model(inputs=[original_input, mask_weights, true_masks], outputs=y_pred)
#
#     return model

def get_VGGunet(input_shape=(416, 416, 3), dropout_val=0.1, batch_norm=False, n_filters=32):
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(input_shape),
                        pooling=None, classes=1)
    for l in vgg_model.layers:
        l.trainable = False
    conv1 = vgg_model.get_layer("block1_conv2").output
    conv2 = vgg_model.get_layer("block2_conv2").output
    conv3 = vgg_model.get_layer("block3_conv3").output
    conv4 = vgg_model.get_layer("block4_conv3").output
    conv5 = vgg_model.get_layer("block5_conv3").output

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = double_conv_layer(up1, n_filters*8, dropout_val, batch_norm)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = double_conv_layer(up2, n_filters*4, dropout_val, batch_norm)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = double_conv_layer(up3, n_filters*2, dropout_val, batch_norm)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = double_conv_layer(up4, n_filters*1, 0, batch_norm)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='output')(conv9)

    model = Model(input=vgg_model.input, output=conv10)
    return model

def get_VGGunet_bnd(input_shape=(416, 416, 3), dropout_val=0.1, batch_norm=False, n_filters=32, classes=2):
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape,
                            pooling=None, classes=1)
    for l in vgg_model.layers:
        l.trainable = False
    conv1 = vgg_model.get_layer("block1_conv2").output
    conv2 = vgg_model.get_layer("block2_conv2").output
    conv3 = vgg_model.get_layer("block3_conv3").output
    conv4 = vgg_model.get_layer("block4_conv3").output
    conv5 = vgg_model.get_layer("block5_conv3").output

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = double_conv_layer(up1, n_filters*8*2, dropout_val, batch_norm)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = double_conv_layer(up2, n_filters*8, dropout_val, batch_norm)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = double_conv_layer(up3, n_filters*4, dropout_val, batch_norm)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = double_conv_layer(up4, n_filters*2, 0, batch_norm)

    conv10 = Conv2D(classes, (1, 1), activation='sigmoid', name='output')(conv9)

    model = Model(input=vgg_model.input, output=conv10)

    return model



def get_VGGunet_bnd_v2(input_shape=(416, 416, 3), dropout_val=0.1, batch_norm=False, n_filters=32, classes=2):
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape,
                            pooling=None, classes=1)
    for l in vgg_model.layers:
        l.trainable = False
    conv1 = vgg_model.get_layer("block1_conv2").output
    conv2 = vgg_model.get_layer("block2_conv2").output
    conv3 = vgg_model.get_layer("block3_conv3").output
    conv4 = vgg_model.get_layer("block4_conv3").output
    conv5 = vgg_model.get_layer("block5_conv3").output

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = double_conv_layer_v2(up1, n_filters*8*4, dropout_val, batch_norm)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = double_conv_layer_v2(up2, n_filters*8*3, dropout_val, batch_norm)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = double_conv_layer_v2(up3, n_filters*8*2, dropout_val, batch_norm)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = double_conv_layer_v2(up4, n_filters*8, 0, batch_norm)

    conv10 = double_conv_layer_v2(conv9, n_filters*4, 0, batch_norm)

    out = Conv2D(classes, (1, 1), activation='sigmoid', name='output')(conv10)

    model = Model(input=vgg_model.input, output=out)

    return model

def get_VGGunet_bnd_v3(input_shape=(416, 416, 3), dropout_val=0.1, batch_norm=False, n_filters=32, classes=2):
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape,
                            pooling=None, classes=1)
    for l in vgg_model.layers:
        l.trainable = False

    conv1 = vgg_model.get_layer("block1_conv2").output
    conv2 = vgg_model.get_layer("block2_conv2").output
    conv3 = vgg_model.get_layer("block3_conv3").output
    conv4 = vgg_model.get_layer("block4_conv3").output
    conv5 = vgg_model.get_layer("block5_conv3").output

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = double_conv_layer(up1, n_filters*8*3, dropout_val, batch_norm)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = double_conv_layer(up2, n_filters*8*2, dropout_val, batch_norm)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = double_conv_layer(up3, n_filters*8*1, dropout_val, batch_norm)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = double_conv_layer(up4, n_filters*4, 0, batch_norm)

    out = Conv2D(classes, (1, 1), activation='sigmoid', name='output')(conv9)

    model = Model(input=vgg_model.input, output=out)

    return model

# ================================ don't scroll down ===================================

