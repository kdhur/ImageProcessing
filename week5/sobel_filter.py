import cv2
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering

def get_sobel():
    derivative = np.array([[-1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivative)
    y = np.dot(derivative.T, blur.T)

    return x, y

def main():
    sobel_x, sobel_y = get_sobel()

    src = cv2.imread('../imgs/sobel_test.png', cv2.IMREAD_GRAYSCALE)
    dst_x = my_filtering(src, sobel_x)
    dst_y = my_filtering(src, sobel_y)

    # dst_x = np.clip(dst_x, 0, 255).astype(np.uint8)
    # dst_y = np.clip(dst_y, 0, 255).astype(np.uint8) #-> 이걸 안하면 밑에 /255를 해야 float로 표현 가능
    #절댓값
    dst_x = np.abs(dst_x)
    dst_y = np.abs(dst_y)

    dst = dst_x + dst_y

    ret, dst_threshold = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('before_threshold', dst/255)
    cv2.imshow('after_threshold', dst_threshold/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()