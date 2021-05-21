import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):
    h, w = src.shape[:2] #src를 2차원 배열로 만들고 행,열값 h, w에 저장
    hist = np.zeros((256,), dtype=int) #maxgraylevel만큼 0으로 채운 배열
    for row in range(h):
        for col in range(w):
            intensity = src[row, col] #좌표마다 픽셀을 intensity에 저장
            hist[intensity] += 1 #픽셀값에 해당하는 index에 +1
    ###############################
    # TODO                        #
    # my_calcHist완성             #
    # src : input image           #
    # hist : src의 히스토그램      #
    ###############################
    return hist

def my_normalize_hist(hist, pixel_num):
    normalized_hist = np.zeros((len(hist),)) #hist만큼 길이의 0으로 채운 1차원 배열 생성
    for i in range(len(hist)):
        normalized_hist[i] = hist[i] / pixel_num #히스토그램값을 총 픽셀수로 나눠서 배열에 저장
    ########################################################
    # TODO                                                 #
    # my_normalize_hist완성                                #
    # hist : 히스토그램                                     #
    # pixel_num : image의 전체 픽셀 수                      #
    # normalized_hist : 히스토그램값을 총 픽셀수로 나눔      #
    ########################################################
    return normalized_hist


def my_PDF2CDF(pdf):
    cdf = np.zeros((len(pdf),))
    cdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cdf[i] = cdf[i - 1] + pdf[i] #반복문으로 누적합을 구함
    ########################################################
    # TODO                                                 #
    # my_PDF2CDF완성                                       #
    # pdf : normalized_hist                                #
    # cdf : pdf의 누적                                     #
    ########################################################
    return cdf


def my_denormalize(normalized, gray_level):
    denormalized = np.zeros((len(normalized),))
    for i in range(len(normalized)):
        denormalized[i] = gray_level * normalized[i] #normalized와 gray_level을 곱함
    ########################################################
    # TODO                                                 #
    # my_denormalize완성                                   #
    # normalized : 누적된pdf값(cdf)                        #
    # gray_level : max_gray_level                          #
    # denormalized : normalized와 gray_level을 곱함        #
    ########################################################
    return denormalized


def my_calcHist_equalization(denormalized, hist):
    hist_equal = np.zeros((len(denormalized),))

    for i in range(len(denormalized)):
        for j in range(len(denormalized)):
            if denormalized[j] == i : #demormalized 값과  0부터의 값이 같으면
                hist_equal[i] += hist[j] #같은 값에 해당하는 인덱스에 histogram 값을 더해줌

    ###################################################################
    # TODO                                                            #
    # my_calcHist_equalization완성                                    #
    # denormalized : output gray_level(정수값으로 변경된 gray_level)   #
    # hist : 히스토그램                                                #
    # hist_equal : equalization된 히스토그램                           #
    ####################################################################
    return hist_equal


def my_equal_img(src, output_gray_level):
    h, w = src.shape[:2]  # src를 2차원 배열로 만들고 행,열값 h, w에 저장
    dst = np.zeros((h,w), dtype=np.uint8)  # src와 같은 크기인 0의 배열 생성
    for row in range(h):
        for col in range(w):
            intensity = src[row, col]  # 좌표마다 픽셀을 intensity에 저장
            if src[row, col] != output_gray_level[intensity]:
                dst[row, col] = output_gray_level[intensity]  # 픽셀값에 해당하는 index에 +1
    ###################################################################
    # TODO                                                            #
    # my_equal_img완성                                                #
    # src : input image                                               #
    # output_gray_level : denormalized(정수값으로 변경된 gray_level)   #
    # dst : equalization된 결과 이미지                                 #
    ####################################################################
    return dst

#input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram) #여기부터 이상
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    # show mapping function
    ###################################################################
    # TODO                                                            #
    # plt.plot(???, ???)완성                                           #
    # plt.plot(y축, x축)                                               #
    ###################################################################
    input_gray_level = np.zeros((len(output_gray_level),))
    for i in range(len(output_gray_level)) :
        input_gray_level[i] = i #input_gray_level을 0부터 255까지 채움
    plt.plot(input_gray_level, output_gray_level)
    plt.title('mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

if __name__ == '__main__':
    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    plt.figure(figsize=(8, 5))
    cv2.imshow('equalization after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

