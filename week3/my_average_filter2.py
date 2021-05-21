import cv2
import numpy as np

def my_average_filter_3x3(src):
    mask = np.array([[1/12, 1/12, 1/12],
                     [1/12, 1/12, 1/12],
                     [1/12, 1/12, 1/12]]) #총합이 1보다 작기 때문에 전체 이미지가 어두워짐 #픽셀값 x 9/12(픽셀값이 같은 경우)
    dst = cv2.filter2D(src, -1, mask)
    return dst

if __name__=='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_average_filter_3x3(src)

    cv2.imshow('orginal', src)
    cv2.imshow('average filter', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()