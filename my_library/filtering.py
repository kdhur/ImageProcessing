import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.padding import my_padding


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (row, col) = mask.shape
    src_pad = my_padding(src, (row // 2, col // 2), pad_type)
    dst = np.zeros((h, w))

    for i in range(row // 2, h + row // 2):
        for j in range(col // 2, w + col // 2):
            target = src_pad[i - row // 2: i + (row // 2 + 1), j - col // 2: j + (col // 2 + 1)]
            dst[i - row // 2][j - col // 2] = np.sum(target * mask)

    return dst

def my_filtering_TA(src, filter, pad_type='zero'): #TA_solution
    (h, w) = src.shape
    (f_h, f_w) = filter.shape
    src_pad = my_padding(src, (f_h // 2, f_w // 2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w] * filter)
            dst[row, col] = val

    return dst
