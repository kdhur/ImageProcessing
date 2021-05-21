import cv2
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.mask import get_DoG_filter
from my_library.filtering import my_filtering_TA


# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################
    DoG_x, DoG_y = get_DoG_filter(fsize, sigma)
    Ix = my_filtering_TA(src, DoG_x, 'zero')
    Iy = my_filtering_TA(src, DoG_y, 'zero')
    return Ix, Iy


# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):  # 입력이 육십분법인지 호도법인지 알아보고 하기
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    angle = np.arctan(Iy / (Ix + e))
    return angle


# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    (x, y) = magnitude.shape
    largest_magnitude = np.zeros((x, y))

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            if 0 <= angle[i][j] < np.pi / 4:
                p = (np.tan(angle[i][j]) * magnitude[i - 1][j - 1]) + \
                    ((1 - np.tan(angle[i][j])) * magnitude[i][j - 1])
                r = (np.tan(angle[i][j]) * magnitude[i + 1][j + 1]) + \
                    ((1 - np.tan(angle[i][j])) * magnitude[i][j + 1])
            elif np.pi / 4 <= angle[i][j] < np.pi / 2:
                p = ((np.tan((np.pi / 2) - angle[i][j])) * magnitude[i - 1][j - 1]) + \
                     ((1 - (np.tan((np.pi / 2) - angle[i][j]))) * magnitude[i - 1][j])
                r = ((np.tan((np.pi / 2) - angle[i][j])) * magnitude[i + 1][j + 1]) + \
                    ((1 - (np.tan((np.pi / 2) - angle[i][j]))) * magnitude[i + 1][j])
            elif -np.pi / 4 <= angle[i][j] < 0:
                p = (np.tan(-angle[i][j]) * magnitude[i - 1][j + 1]) + \
                    ((1 - np.tan(-angle[i][j])) * magnitude[i][j + 1])
                r = (np.tan(-angle[i][j]) * magnitude[i + 1][j - 1]) + \
                    ((1 - np.tan(-angle[i][j])) * magnitude[i][j - 1])
            elif -np.pi / 2 < angle[i][j] < -np.pi / 4:
                p = ((np.tan((np.pi / 2) + angle[i][j])) * magnitude[i - 1][j + 1]) + \
                    ((1 - (np.tan((np.pi / 2) + angle[i][j]))) * magnitude[i - 1][j])
                r = ((np.tan((np.pi / 2) + angle[i][j])) * magnitude[i + 1][j - 1]) + \
                    ((1 - (np.tan((np.pi / 2) + angle[i][j]))) * magnitude[i + 1][j])

            if magnitude[i][j] > p and magnitude[i][j] > r:
                largest_magnitude[i][j] = magnitude[i][j]
            else:
                largest_magnitude[i][j] = 0

    return largest_magnitude


# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy()

    # dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)

    (h, w) = dst.shape
    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################
    for i in range(h):
        for j in range(w):
            if dst[i][j] >= high_threshold_value:
                dst[i][j] = 255
            elif dst[i][j] < low_threshold_value:
                dst[i][j] = 0
            else:
                dst[i][j] = 127

    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    for i in range(h):
        for j in range(w):
            if dst[i][j] == 255:
                stack = [[i, j]]
                while stack:  # 스택에 남은것이 없을 때까지 반복
                    var = stack.pop()
                    x = var[0]
                    y = var[1]
                    for k in range(8):
                        nx = x + dx[k]
                        ny = y + dy[k]
                        if dst[nx][ny] == 127:
                            dst[nx][ny] = 255
                            stack.append([nx, ny])

    for i in range(h):
        for j in range(w):
            if dst[i][j] == 127:
                dst[i][j] = 0

    return dst


def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
