import cv2

def main():
    src = cv2.imread('../imgs/threshold_test.png', cv2.IMREAD_GRAYSCALE)
    #ret, dst = cv2.threshold(src, 150, 255, cv2.THRESH_BINARY)
    ret, dst = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU) #OTSU-> 자동으로 threshold 값을 정해줌

    print('ret : ', ret)
    cv2.imshow('orginal', src)
    cv2.imshow('threshold_test', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()