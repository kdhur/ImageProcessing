import cv2
import numpy as np
#데이터타입 uint8이 아니라 float로 바꾸고 연산 후 마지막에 0~255값으로 바꿔줘야 정상적으로 나옴
#(sharpening filter 오버플로 문제)
def my_padding(src, pad_shape, pad_type='zero'):
    #zero padding
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up 윗부분 복사해서 채우기
        for i in range(w+2*p_w) :
            pad_img[:p_h, i] = pad_img[p_h, i]
        #down
        for i in range(w+2*p_w) :
            pad_img[(h+2*p_h - 1) - p_h:, i] = pad_img[(h+2*p_h - 1) - p_h,i]
        #left
        for i in range(h+2*p_h) :
            pad_img[i, :p_w] = pad_img[i, p_w]
        #right
        for i in range(w+2*p_h) :
            pad_img[i, (w+2*p_w - 1) - p_w:] = pad_img[i, (w+2*p_w - 1) - p_w]

    else:
        print('zero padding')

    return pad_img

def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
    dst = np.zeros((h, w))
    row, col = fshape

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################
        mask = np.full(fshape, 1 / (row * col))

        #mask 확인
        # print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################
        v1 = np.zeros(fshape)
        v1[row//2, col//2] = 2
        v2 = np.full(fshape, 1 / (row * col))
        mask = v1 - v2

        #mask 확인
        # print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################

    # No padding
    if (pad_type == 'X'):
        for i in range(fshape[0] // 2, h - fshape[0] // 2):
            for j in range(fshape[1] // 2, w - fshape[1] // 2):
                target = src[i - fshape[0] // 2: i + (fshape[0] // 2 + 1),
                             j - fshape[1] // 2: j + (fshape[1] // 2 + 1)]
                if (np.sum(target * mask) > 255):
                    dst[i][j] = 255
                elif (np.sum(target * mask) < 0):
                    dst[i][j] = 0
                else:
                    dst[i][j] = np.sum(target * mask)
        dst = (dst + 0.5).astype(np.uint8)

        return dst


    for i in range(fshape[0] // 2, h + fshape[0] // 2):
        for j in range(fshape[1] // 2, w + fshape[1] // 2):
            target = src_pad[i - fshape[0]//2 : i + (fshape[0]//2 + 1), j - fshape[1]//2 : j + (fshape[1]//2 + 1)]
            if (np.sum(target * mask) > 255) :
                dst[i - fshape[0]//2][j - fshape[1]//2] = 255
            elif (np.sum(target * mask) < 0) :
                dst[i - fshape[0] // 2][j - fshape[1] // 2] = 0
            else :
                dst[i - fshape[0] // 2][j - fshape[1] // 2] = np.sum(target * mask)

    dst = (dst+0.5).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    rep_test = my_padding(src, (20,20), 'repetition')

    # 3x3 filter
    dst_average_3x3 = my_filtering(src, 'average', (3,3))
    dst_sharpening_3x3 = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정
    my_dst_average = my_filtering(src, 'average', (19,19))
    my_dst_sharpening = my_filtering(src, 'sharpening', (19,19))

    # 11x13 filter
    dst_average_11x13 = my_filtering(src, 'average', (11,13), 'repetition')
    dst_sharpening_11x13 = my_filtering(src, 'sharpening', (11,13), 'repetition')

    cv2.imshow('original', src)
    cv2.imshow('3x3 average filter', dst_average_3x3)
    cv2.imshow('3x3 sharpening filter', dst_sharpening_3x3)
    cv2.imshow('my average filter', my_dst_average)
    cv2.imshow('my sharpening filter', my_dst_sharpening)
    cv2.imshow('11x13 average filter', dst_average_11x13)
    cv2.imshow('11x13 sharpening filter', dst_sharpening_11x13)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
