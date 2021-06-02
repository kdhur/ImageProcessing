import numpy as np
import cv2
import time


def C(w, n=8):
    if w == 0:
        return (1 / n) ** 0.5
    else:
        return (2 / n) ** 0.5


def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance


def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################
    h, w = src.shape
    blocks = []

    for i in range(h // n):
        for j in range(w // n):
            blocks.append(src[i * n: n + i * n, j * n: j * n + n])

    return np.array(blocks, float)


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    dst = np.zeros(block.shape)
    v, u = dst.shape

    y, x = np.mgrid[0:u, 0:v]

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * np.cos(((2 * x + 1) * u_ * np.pi) / (2 * n)) * np.cos(((2 * y + 1) * v_ * np.pi) / (2 * n))
            dst[v_, u_] = C(u_) * C(v_) * np.sum(tmp)

    return np.round(dst)


def my_zigzag_scanning(block, mode='encoding', block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################
    if mode == 'encoding':
        h, w = block.shape
        zigzag = []
        cnt = 1
        cX = 0
        cY = 0
        zigzag.append('EOB')
        block = np.flip(block)

        while cnt < block_size * block_size:
            if cY + 1 < block_size:
                cY += 1
            else:
                cX += 1
            if block[cX][cY] != 0:
                zigzag.append(block[cX][cY])
            elif len(zigzag) > 1:
                zigzag.append(block[cX][cY])
            cnt += 1

            while cY - 1 > -1 and cX + 1 < block_size:
                cX += 1
                cY -= 1
                if block[cX][cY] != 0:
                    zigzag.append(block[cX][cY])
                elif len(zigzag) > 1:
                    zigzag.append(block[cX][cY])
                cnt += 1

            if cX + 1 < h:
                cX += 1
            else:
                cY += 1
            if block[cX][cY] != 0:
                zigzag.append(block[cX][cY])
            elif len(zigzag) > 1:
                zigzag.append(block[cX][cY])
            cnt += 1

            while cX - 1 > -1 and cY + 1 < w:
                cX -= 1
                cY += 1
                if block[cX][cY] != 0:
                    zigzag.append(block[cX][cY])
                elif len(zigzag) > 1:
                    zigzag.append(block[cX][cY])
                cnt += 1

        zigzag.reverse()
        return np.array(zigzag)

    else:
        dst = np.zeros((block_size, block_size))
        cnt = 1
        cX = 0
        cY = 0
        if block[0] == 'EOB':
            return dst

        dst[0][0] = block[0]
        while 1:
            if cX + 1 < block_size:
                if block[cnt] == 'EOB':
                    break
                cX += 1
            elif cY + 1 < block_size:
                if block[cnt] == 'EOB':
                    break
                cY += 1

            if block[cnt] == 'EOB':
                break
            dst[cX][cY] = block[cnt]
            cnt += 1

            while cX - 1 > -1 and cY + 1 < block_size:
                if block[cnt] == 'EOB':
                    break
                cX -= 1
                cY += 1
                dst[cX][cY] = block[cnt]
                cnt += 1

            if cY + 1 < block_size:
                if block[cnt] == 'EOB':
                    break
                cY += 1
            elif cX + 1 < block_size:
                if block[cnt] == 'EOB':
                    break
                cX += 1
            if block[cnt] == 'EOB':
                break
            dst[cX][cY] = block[cnt]
            cnt += 1

            while cY - 1 > -1 and cX + 1 < block_size:
                if block[cnt] == 'EOB':
                    break
                cX += 1
                cY -= 1
                dst[cX][cY] = block[cnt]
                cnt += 1

        return dst


def DCT_inv(block, n=8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    dst = np.zeros((n, n))

    v, u = np.mgrid[0:n, 0:n]
    cUV = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cUV[i][j] = C(v[i][j]) * C(u[i][j])

    for x in range(n):
        for y in range(n):
            tmp = block * cUV * np.cos(((2 * x + 1) * u * np.pi) / 16) * np.cos(((2 * y + 1) * v * np.pi) / 16)
            dst[x][y] = np.sum(tmp)

    return np.round(dst)


def block2img(blocks, src_shape, n=8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    h, w = src_shape
    dst = np.zeros((h, w))
    cnt = 0

    for i in range(h // n):
        for j in range(w // n):
            dst[i * n: n + i * n, j * n: j * n + n] = blocks[cnt]
            cnt += 1

    dst = np.clip(dst, 0, 255)
    dst = dst.astype(np.uint8)
    return dst


def Encoding(src, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)

    # subtract 128
    blocks -= 128
    # DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    # Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))

    return zz, src.shape


def Decoding(zigzag, src_shape, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    # add 128
    blocks_idct += 128

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst


def main():
    start = time.time()
    # src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    # comp, src_shape = Encoding(src, n=8)

    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 2줄은 주석처리하고, 아래 2줄은 주석 풀기)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')

    recover_img = Decoding(comp, src_shape, n=8)
    total_time = time.time() - start

    print('time : ', total_time)
    if total_time > 45:
        print('감점 예정입니다.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
