import cv2
import numpy as np

class Normalize(object):
    def __init__(self, norm_type, mean_std=None):
        assert norm_type in ['imagenet', 'clahe', 'both', 'mean_std']
        self.norm_type = norm_type
        if mean_std is None:
            self.mean = np.array([0.44, 0.46, 0.46])
            self.std = np.array([0.16, 0.15, 0.15])
        else:
            self.mean = mean_std[0]
            self.std = mean_std[1]

    def imagenet_norm(self, img):
        img = img / 255.
        img = (img[:, :, [2, 1, 0]] - self.mean)
        img = img.astype(np.float32)
        return img

    def CLAHE_norm(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm_mean = lambda x: 255 - x if x.mean() > 127 else x
        return (norm_mean(clahe.apply(lab[:, :, 0]))).astype(np.float32)

    def mean_std(self, img):
        img = img / 255.
        img = (img[:, :, [2, 1, 0]] - self.mean) / self.std
        img = img.astype(np.float32)
        return img

    def both_norm(self, img):
        img_1 = self.imagenet_norm(img)
        clahe_img = self.CLAHE_norm(img)
        r_channel, g_channel, b_channel = cv2.split(img_1)
        return cv2.merge((r_channel, g_channel, b_channel, clahe_img))

    def __call__(self, img):
        if self.norm_type == 'imagenet':
            return self.imagenet_norm(img)
        elif self.norm_type == 'clahe':
            return self.CLAHE_norm(img)
        elif self.norm_type == 'both':
            return self.both_norm(img)
        elif self.norm_type == 'mean_std':
            return self.mean_std(img)

