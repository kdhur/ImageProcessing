import numpy as np
import cv2
import time
import math

# library add
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.padding import my_padding


def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-msize//2.0+1:msize//2.0+1, -msize//2.0+1:msize//2.0+1]

    # temp = np.zeros(msize)
    # for i in range(-msize//2+1, msize//2+1) :
    #     temp[i + msize//2] = i
    # x = np.full((msize, msize), temp)

    gaus2D = x**2 + y**2


    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''

    # 2차 gaussian mask 생성
    for i in range(msize) :
        for j in range(msize) :
            gaus2D[i][j] = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(gaus2D[i][j] / (2 * (sigma ** 2))))

    # mask의 총 합 = 1 -> 괄호 잘 확인하기
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기 -> 2차원 배열로 선언하면 편함 배열.T를 하면 (5,1)을 (1,5)로 바꾸기 쉬움
    #########################################
    temp = np.zeros(msize)
    gaus1D = np.zeros((1, msize))
    for i in range(-msize//2+1, msize//2+1) :
        temp[i + msize//2] = i
    x = np.full((1, msize), temp**2)

    '''
    x = np.full((1, 3), [-1, 0, 1])
    x = [[ -1, 0, 1]]

    x = np.array([[-1, 0, 1]])
    x = [[ -1, 0, 1]]
    '''
    for i in range(msize) :
            gaus1D[0][i] = (1 / math.sqrt(2 * math.pi) * sigma) * math.exp(-(x[0][i]/(2*(sigma**2))))

    # mask의 총 합 = 1
    gaus1D /= np.sum(gaus1D)
    return gaus1D


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    print('<mask>')
    print(mask)

    # 시간을 측정할 때 만 이 코드를 사용하고 시간측정 안하고 filtering을 할 때에는
    # 4중 for문으로 할 경우 시간이 많이 걸리기 때문에 2중 for문으로 사용하기. 4중 for문일때만 1D가 더 빠름.
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            sum = 0
            for m_row in range(m_h):
                for m_col in range(m_w):
                    sum += pad_img[row + m_row, col + m_col] * mask[m_row, m_col]
            dst[row, col] = sum

    return dst


if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    mask_size = 5
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma=1)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma=1)

    print('mask size : ', mask_size)
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus1D = my_filtering(src, gaus1D.T)
    dst_gaus1D = my_filtering(dst_gaus1D, gaus1D)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end - start)

    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus2D = my_filtering(src, gaus2D)
    end = time.perf_counter()  # 시간 측정 끝
    print('2D time : ', end - start)

    dst_gaus2D = cv2.filter2D(src, -1, gaus2D)

    dst_gaus1D = np.clip(dst_gaus1D + 0.5, 0, 255)
    dst_gaus1D = dst_gaus1D.astype(np.uint8)
    dst_gaus2D = np.clip(dst_gaus2D + 0.5, 0, 255)
    dst_gaus2D = dst_gaus2D.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('1D gaussian img', dst_gaus1D)
    cv2.imshow('2D gaussian img', dst_gaus2D)
    cv2.waitKey()
    cv2.destroyAllWindows()