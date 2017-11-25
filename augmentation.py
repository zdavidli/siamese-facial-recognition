from skimage import transform as tf
import numpy as np
from PIL import Image

def augmentation(im, augmentation_prob=0.7):
    #parameters
    #  im - PIL image
    
    if np.random.random_sample() < augmentation_prob:
        x = np.random.random_sample()
        if x < 0.25:
            return lr(im)
        elif x < 0.5:
            return rotate(im)
        elif x < 0.75:
            return translate(im)
        else:
            return zoom(im)
    else:
        return im
    
def lr(im):
    return Image.fromarray(np.fliplr(np.array(im)))

def rotate(im):
    return Image.fromarray((255* tf.rotate(np.array(im), np.random.randint(-30, 31), mode='constant')).astype('uint8'))

def translate(im):
    translate = tf.AffineTransform(translation=
                                           (np.random.randint(-10, 11), np.random.randint(-10, 11))
                                          )
    return Image.fromarray((255*tf.warp(im, translate, mode='constant')).astype('uint8'))

def zoom(im):
    ratio = 0.7+0.6*np.random.random()
    scaled = tf.rescale(np.array(im), ratio, mode='constant')
    s, _,_ = scaled.shape

    if ratio >= 1:
        left = s//2 - 125
        right = s//2 + 125
        bottom = s//2 - 125
        top = s//2 + 125
        scaled = scaled[left:right, bottom:top]
    else:
        pad = (250-s)//2
        scaled = np.lib.pad(scaled, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        scaled = tf.resize(scaled, (250,250), mode='constant')

    return Image.fromarray((255*scaled).astype('uint8'))

