import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy

def dice_coef_weighted_one_class(y_true, y_pred):
    weight = y_true[:, :, :, 1:]
    y_true = y_true[:, :, :, :1]
    intersection = K.sum(y_true * y_pred)
    smooth = K.epsilon()
    return -(2. * intersection + smooth) * weight / (K.sum(y_true) + K.sum(y_pred) + smooth )

def dice_coef_weighted_metric_one_class(y_true, y_pred):
    weight = y_true[:, :, :, 1:]
    y_true = y_true[:, :, :, :1]
    intersection = K.sum(y_true * y_pred)
    smooth = K.epsilon()
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_weighted(y_true, y_pred):
    print(K.int_shape(y_pred))
    weight = K.repeat_elements(y_true[:, :, :, 2:], 2, axis=3)
    # print(K.int_shape(weight))
    y_true = y_true[:, :, :, :2]
    print(K.shape(y_true), K.shape(y_pred))
    intersection = K.sum(y_true * y_pred)
    smooth = K.epsilon()
    return (2. * intersection + smooth) * weight / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_weighted_loss(y_true, y_pred):
    return -K.log(dice_coef_weighted(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = K.epsilon() #1.
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_dice_and_binary_loss(X):
    y_pred, y_true, weights = X
    dice_loss = -K.log(dice_coef(y_true, y_pred))
    binary_loss = K.binary_crossentropy(y_true, y_pred)
    binary_loss = binary_loss * weights
    return binary_loss + dice_loss*4

def dice_coef_and_binary_loss(y_true, y_pred): #was *4 1*
    return -K.log(dice_coef(y_true, y_pred))*4 + 1*binary_crossentropy(y_true, y_pred)

def identity_loss(y_true, y_pred):
    return y_pred

def mean_iou(y_true, y_pred):
    import numpy as np
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec))

def weighted_binary_loss(X):
    from zoo_losses_K import dice_coef
    import keras.backend as K
    from keras.layers import multiply
    y_pred, weights, y_true = X
    dice_loss = -K.log(dice_coef(y_true, y_pred))
    binary_loss = K.binary_crossentropy(y_true, y_pred)
    binary_loss = multiply([binary_loss, weights])
    return K.mean(binary_loss, axis=-1, keepdims=True) + 2*dice_loss

import tensorflow as tf

def angularErrorTotal(pred, gt, weight=1, ss=1, outputChannels=1):
    with tf.name_scope("angular_error"):
        # pred = tf.reshape(pred, (-1, outputChannels))
        # gt = tf.to_float(tf.reshape(gt, (-1, outputChannels)))
        # weight = tf.to_float(tf.reshape(weight, (-1, 1)))
        # ss = tf.to_float(tf.reshape(ss, (-1, 1)))

        pred = tf.nn.l2_normalize(pred, 1) * 0.999999
        gt = tf.nn.l2_normalize(gt, 1) * 0.999999

        errorAngles = tf.acos(tf.reduce_sum(pred * gt, reduction_indices=[1], keepdims=True))

        lossAngleTotal = tf.reduce_sum((tf.abs(errorAngles*errorAngles)))#*ss*weight)

        return lossAngleTotal / 1000

def countTotal(ss):
    with tf.name_scope("total"):
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))
        total = tf.reduce_sum(ss)

        return total

def angularErrorLoss(pred, gt, weight, ss, outputChannels=2):
        lossAngleTotal = angularErrorTotal(pred=pred, gt=gt, ss=ss, weight=weight, outputChannels=outputChannels) \
                         / (countTotal(ss)+1)

        tf.add_to_collection('losses', lossAngleTotal)

        totalLoss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return totalLoss