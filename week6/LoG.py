import cv2
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering

def get_LoG_filter(fsize, sigma=1):
    y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]

    LoG = (1 / (2*np.pi * sigma**2)) * (((x**2 + y**2) / sigma**4) - (2 / sigma**2)) \
          * np.exp(-((x**2 + y**2) / (2 * sigma**2)))

    LoG = LoG - (LoG.sum() / fsize**2)

    return np.round(LoG, 3)

def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    src = src / 255
    Log_filter = get_LoG_filter(fsize=9, sigma=1)
    print(Log_filter)

    dst = my_filtering(src, Log_filter, 'zero')
    print(dst.max(), dst.min())

    dst = np.abs(dst)
    dst = dst - dst.min()
    dst = dst / dst.max()
    print(dst.max(), dst.min()) #0~1 사이로 만들어줌

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
