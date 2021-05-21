import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    #zero padding
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h,p_w:p_w+w] = src

    return pad_img