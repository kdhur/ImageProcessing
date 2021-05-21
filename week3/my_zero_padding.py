import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    #zero padding
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h,p_w:p_w+w] = src

    return pad_img

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    dst1 = my_padding(src, (20,20)) #20,20이 검은 부분 크기
    dst1 = dst1.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('zero padding', dst1)
    cv2.waitKey()
    cv2.destroyAllWindows()
