import cv2
import numpy as np

def my_average_filter_3x3(src):
    #mask = np.array([[-1/9, -1/9, -1/9],
    #                 [-1/9, 26/9, -1/9],
    #                 [-1/9, -1/9, -1/9]]) #총합이 1보다 크기 때문에 선명해지면서 밝아짐
    v1 = np.array([[0, 0, 0],
                   [0, 3, 0],
                   [0, 0, 0]])
    v2 = np.array([[1 / 9, 1 / 9, 1 / 9],
                   [1 / 9, 1 / 9, 1 / 9],
                   [1 / 9, 1 / 9, 1 / 9]])
    mask = v1 - v2
    dst = cv2.filter2D(src, -1, mask)
    return dst

if __name__=='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_average_filter_3x3(src)

    cv2.imshow('orginal', src)
    cv2.imshow('average filter', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()