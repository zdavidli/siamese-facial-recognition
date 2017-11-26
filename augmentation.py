from skimage import transform as tf
import numpy as np
from PIL import Image

def augmentation(im, augmentation_prob=0.7):
    """
    Parameters
        im - Input PIL image
        augmentation_prob - probability of applying augmentations
    Return value
        im - Augmented PIL Image
        
    Notes
        If we randomly choose to augment, may actually not augment, since each
        augmentation has 80% chance of applying.
        
        Chance of not applying augmentation is when entering first if statement
        is 0.2^4 = .16%
    """
    
    if np.random.random_sample() < augmentation_prob:
        if np.random.random() < 0.8:
            im = lr(im)
        if np.random.random() < 0.8:
            im = rotate(im)
        if np.random.random() < 0.8:
            im = translate(im)
        if np.random.random() < 0.8:
            im = zoom(im)
    return im

"""
Left-right flip
"""
def lr(im):
    return Image.fromarray(np.fliplr(np.array(im)))
"""
Random rotation of -30 degrees to 30 degrees
"""
def rotate(im):
    return Image.fromarray((255* tf.rotate(np.array(im), np.random.randint(-30, 31), mode='constant')).astype('uint8'))

"""
Random translation of -10 to 10 pixels in vertical and horizontal direction
"""
def translate(im):
    translate = tf.AffineTransform(translation=
                                           (np.random.randint(-10, 11), np.random.randint(-10, 11))
                                          )
    return Image.fromarray((255*tf.warp(im, translate, mode='constant')).astype('uint8'))

"""
Random zoom of ratio 0.7 to 1.3. 
"""
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
        
        #Need to rescale to 250x250 because integer division will cause scaled to be 249x249
        scaled = tf.resize(scaled, (250,250), mode='constant')

    return Image.fromarray((255*scaled).astype('uint8'))

