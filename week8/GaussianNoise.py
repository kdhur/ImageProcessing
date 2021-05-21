import numpy as np
import cv2

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.padding import my_padding
from my_library.filtering import my_filtering_TA

def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255) #0보다 작거나 255보다 큰 경우 방지
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):
    dst = src / 255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return my_normalize(dst)

def main():
    msize = 5
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    mask = np.zeros((msize, msize))
    sigma = 0.1
    sigma_r = 0.2
    cnt = 0
    dx = np.zeros(msize ** 2)
    dy = np.zeros(msize ** 2)
    for j in range(-msize // 2 + 1, msize // 2 + 1):
        for k in range(msize):
            dx[cnt] = j
            dy[cnt] = (-msize // 2 + 1) + k
            cnt = cnt + 1
    h, w = src.shape
    for i in range(h):
        for j in range(w):
            for k in range(msize**2):
                ni = int((msize//2) + dx[k])
                nj = int((msize//2) + dy[k])

                mask[ni][nj] = \
                np.exp(-((dx[k]**2) / (2*(sigma**2))) - ((dy[k]**2) / (2*(sigma**2)))) *\
                np.exp(-(src[i][j] - src[max(i + dx[k], 0)][max(j + dx[k], 0)]) / (2*(sigma_r**2)))



    np.random.seed(seed=100)
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst_noise = add_gaus_noise(src, mean=0, sigma=0.1)

    #평균 필터로 노이즈 제거
    # mask_size = 5
    # mask = np.ones((mask_size, mask_size)) / (mask_size ** 2)
    #
    # dst = my_filtering_TA(dst_noise, mask)
    # dst = dst.astype(np.uint8)

    h, w = src.shape
    num = 100

    imgs = np.zeros((num, h, w))
    for i in range(num):
        imgs[i] = add_gaus_noise(src, mean=0, sigma=0.1) #서로 다른 가우시안 노이즈가 입혀진 100개의 이미지

    dst = np.mean(imgs, axis=0).astype(np.uint8) #imgs.shape = (100, h, w) 에서 0축인 100을 기준으로 평균을 낸다

    cv2.imshow('original', src)
    cv2.imshow('add gaus noise', dst_noise)
    cv2.imshow('noise removal', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()