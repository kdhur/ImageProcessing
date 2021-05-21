import cv2
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.mask import my_get_Gaussian2D_mask

def main():
    gaus = my_get_Gaussian2D_mask(msize=256, sigma=30)
    print('mask')
    print(gaus)

    gaus = ((gaus - np.min(gaus)) / np.max(gaus - np.min(gaus)) * 255).astype(np.uint8)
    cv2.imshow('gaus', gaus)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()