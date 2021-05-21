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

    src = cv2.imread('../imgs/edge_detection_img.png', cv2.IMREAD_GRAYSCALE)
    dst_x = my_filtering(src, sobel_x)
    dst_y = my_filtering(src, sobel_y)

    #소벨필터의 왼쪽은 음수여서 오른쪽이 양수에서 갑자기 작은 값으로 바뀌어서 필터링된 값이 음수면 0으로 인식하기때문에 한쪽만 결과가 나옴
    #->절댓값 씌워면 됨. np.abs(dst_x)

    cv2.imshow('dst_x', dst_x)
    cv2.imshow('dst_y', dst_y)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()