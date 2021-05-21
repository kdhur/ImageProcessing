import cv2
import numpy as np

def main():
    src = cv2.imread('../imgs/sobel_test.png', cv2.IMREAD_GRAYSCALE)
    src_float = src.astype(np.float32)

    cv2.imshow('uint8', src)
    cv2.imshow('float32', src_float/255)
    #float32로 형변환 후 255로 나눠줘야함. 0~1의 값. 1이상 값 = 1. 음수값 = 0
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()