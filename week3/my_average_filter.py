import cv2
import numpy as np

def my_average_filter_3x3(src):
    mask = np.array([[1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9]])
    dst = cv2.filter2D(src, -1, mask) #ddepth : 이미지 깊이(자료형 크기). -1이면 입력과 동일
    return dst

if __name__=='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_average_filter_3x3(src)

    cv2.imshow('orginal', src)
    cv2.imshow('average filter', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()