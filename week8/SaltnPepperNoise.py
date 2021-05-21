import numpy as np
import cv2
import time

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering_TA

def my_median_filtering(src, msize):
    h, w = src.shape

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            r_start = np.clip(row - msize // 2, 0, h) #가장 끝값 mask가 넘치기 때문에 np.clip 사용
            r_end = np.clip(row + msize // 2, 0, h)

            c_start = np.clip(col - msize // 2, 0, h)
            c_end = np.clip(col + msize // 2, 0, h)
            mask = src[r_start:r_end, c_start:c_end]
            dst[row, col] = np.median(mask)

    return dst.astype(np.uint8)

def add_SnP_noise(src, prob):
    h, w = src.shape

    noise_prob = np.random.rand(h, w) #0과 1사이의 랜덤한 값들의 배열
    dst = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if noise_prob[row, col] < prob:
                dst[row, col] = 0
            elif noise_prob[row, col] > 1 - prob:
                dst[row, col] = 255
            else:
                dst[row, col] = src[row, col]

    return dst

def main():
    np.random.seed(seed=100)
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    snp_noise = add_SnP_noise(src, prob=0.05)

    average_start = time.time()
    mask_size = 5
    mask = np.ones((mask_size, mask_size)) / (mask_size ** 2)

    dst_aver = my_filtering_TA(snp_noise, mask)
    dst_aver = dst_aver.astype(np.uint8)
    print('average filtering time : ', time.time() - average_start) #가우시안 필터도 가능

    median_start = time.time()
    dst_median = my_median_filtering(snp_noise, mask_size)
    print('median filtering time : ', time.time() - median_start)

    cv2.imshow('original', src)
    cv2.imshow('Salt and Pepper noise', snp_noise)
    cv2.imshow('noise removal(average filter)', dst_aver)
    cv2.imshow('noise removal(median filter)', dst_median)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()