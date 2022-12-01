import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.1, r1 = 0.3, method = 'random'):
    #Motivated by https://github.com/Amitayus/Random-Erasing-TensorFlow.git
    #Motivated by https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    img : 3D Tensor data (H,W,Channels) normalized value [0,1]
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    method : 'black', 'white' or 'random'. Erasing type
    -------------------------------------------------------------------------------------
    '''
    assert method in ['random', 'white', 'black'], 'Wrong method parameter'

    if tf.random.uniform([]) > probability:
        return img

    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channels = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)

    target_area = tf.random.uniform([],minval=sl, maxval=sh) * area
    aspect_ratio = tf.random.uniform([],minval=r1, maxval=1/r1)
    h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
    w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

    while tf.constant(True, dtype=tf.bool):
        if h > height or w > width:
            target_area = tf.random.uniform([],minval=sl, maxval=sh) * area
            aspect_ratio = tf.random.uniform([],minval=r1, maxval=1/r1)
            h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)
        else:
            break

    x1 = tf.cond(height == h, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=height - h, dtype=tf.int32))
    y1 = tf.cond(width  == w, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=width - w, dtype=tf.int32))
    
    part1 = tf.slice(img, [0,0,0], [x1,width,channels]) # first row
    part2 = tf.slice(img, [x1,0,0], [h,y1,channels]) # second row 1

    if method is 'black':
        part3 = tf.zeros((h,w,channels), dtype=tf.float32) # second row 2
    elif method is 'white':
        part3 = tf.ones((h,w,channels), dtype=tf.float32)
    elif method is 'random':
        part3 = tf.random.uniform((h,w,channels), dtype=tf.float32)
    
    part4 = tf.slice(img,[x1,y1+w,0], [h,width-y1-w,channels]) # second row 3
    part5 = tf.slice(img,[x1+h,0,0], [height-x1-h,width,channels]) # third row

    middle_row = tf.concat([part2,part3,part4], axis=1)
    img = tf.concat([part1,middle_row,part5], axis=0)

    return img


def color_jitter(image, p=0.5, s=0.1):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.random_brightness(image, max_delta=0.8 * s)
        image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        image = tf.image.random_hue(image, max_delta=0.2 * s)
        image = tf.clip_by_value(image, 0, 1)
    return x

def color_drop(image, p=0.2):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])
    return image

def gaussian_blur(image, sigma_min=0.1, sigma_max=2.0):
    prob = np.random.random_sample()
    if prob < 0.5:
        sigma = (sigma_max - sigma_min) * np.random.random_sample() + sigma_min
        image = tfa.image.gaussian_filter2d(image, (10, 10), sigma)
    return image

@tf.function 
def pair_augmentation(x1, x2, y):
    x1 = tf.map_fn(color_distortion, x1, fn_output_signature=(tf.float32), parallel_iterations=workers)
    x1 = tf.map_fn(gaussian_blur, x1, fn_output_signature=(tf.float32), parallel_iterations=workers)
    x1 = tf.map_fn(random_erasing, x1, fn_output_signature=(tf.float32), parallel_iterations=workers)
    
    x2 = tf.map_fn(color_distortion, x2, fn_output_signature=(tf.float32), parallel_iterations=workers)
    x2 = tf.map_fn(gaussian_blur, x2, fn_output_signature=(tf.float32), parallel_iterations=workers)
    x2 = tf.map_fn(random_erasing, x2, fn_output_signature=(tf.float32), parallel_iterations=workers)
        
    return x1, x2, y

@tf.function
def pair_gaussian_blur(x1, x2, y, **kwargs):
    x1 = gaussian_blur(x1, **kwargs)
    x2 = gaussian_blur(x2, **kwargs)
    return x1, x2, y
    
    
@tf.function
def pair_random_erasing(x1, x2, y, **kwargs):
    x1 = random_erasing(x1, **kwargs)
    x2 = random_erasing(x2, **kwargs)
    return x1, x2, y

@tf.function
def pair_color_drop(x1, x2, y, **kwargs):
    x1 = color_drop(x1, **kwargs)
    x2 = color_drop(x2, **kwargs)
    return x1, x2, y

@tf.function
def pair_color_jitter(x1, x2, y, **kwargs):
    x1 = color_jitter(x1, **kwargs)
    x2 = color_jitter(x2, **kwargs)
    return x1, x2, y



