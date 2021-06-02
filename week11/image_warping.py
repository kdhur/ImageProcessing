import numpy as np
import cv2


def check_size(src, M):
    h, w = src.shape

    P_ori = np.array([
        [0],
        [0],
        [1]
    ])

    P_ori = np.dot(M, P_ori)
    ori_col = P_ori[0][0]
    ori_row = P_ori[1][0]

    P_hw = np.array([
        [w-1],
        [h-1],
        [1]
    ])

    P_hw = np.dot(M, P_hw)
    hw_col = P_hw[0][0]
    hw_row = P_hw[1][0]

    P_a = np.array([
        [0],
        [h-1],
        [1]
    ])

    P_a = np.dot(M, P_a)
    a_col = P_a[0][0]
    a_row = P_a[1][0]

    P_b = np.array([
        [w-1],
        [0],
        [1]
    ])

    P_b = np.dot(M, P_b)
    b_col = P_b[0][0]
    b_row = P_b[1][0]

    maxcol = max(ori_col, hw_col, a_col, b_col)
    maxrow = max(ori_row, hw_row, a_row, b_row)
    mincol = min(ori_col, hw_col, a_col, b_col)
    minrow = min(ori_row, hw_row, a_row, b_row)

    col = maxcol-mincol
    row = maxrow-minrow

    return int(np.ceil(row + 1)), int(np.ceil(col + 1)), int(mincol), int(minrow)


def forward(src, M, fit=False):
    #####################################################
    # TODO                                              #
    # forward 완성                                      #
    #####################################################
    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape
    a, b, minCol, minRow = check_size(src, M)

    if not fit:
        dst = np.zeros((h, w))

    else:
        dst = np.zeros((a, b))
        N = np.zeros(dst.shape)

        for row in range(h):
            for col in range(w):
                P = np.array([
                    [col],
                    [row],
                    [1]
                ])

                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0] - minCol
                dst_row = P_dst[1][0] - minRow

                dst_col_right = int(np.ceil(dst_col))
                dst_col_right = np.clip(dst_col_right, 0, b - 1)
                dst_col_left = int(dst_col)
                dst_col_left = np.clip(dst_col_left, 0, b - 1)

                dst_row_bottom = int(np.ceil(dst_row))
                dst_row_bottom = np.clip(dst_row_bottom, 0, a - 1)
                dst_row_top = int(dst_row)
                dst_row_top = np.clip(dst_row_top, 0, a - 1)

                dst[dst_row_top, dst_col_left] += src[row, col]
                N[dst_row_top, dst_col_left] += 1

                if dst_col_right != dst_col_left:
                    dst[dst_row_top, dst_col_right] += src[row, col]
                    N[dst_row_top, dst_col_right] += 1

                if dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_left] += src[row, col]
                    N[dst_row_bottom, dst_col_left] += 1

                if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_right] += src[row, col]
                    N[dst_row_bottom, dst_col_right] += 1

        dst = np.round(dst / (N + 1E-6))
        dst = dst.astype(np.uint8)
        return dst

    N = np.zeros(dst.shape)

    for row in range(h):
        for col in range(w):
            P = np.array([
                [col],
                [row],
                [1]
            ])

            P_dst = np.dot(M, P)
            dst_col = P_dst[0][0]
            dst_row = P_dst[1][0]

            dst_col_right = int(np.ceil(dst_col))
            dst_col_right = np.clip(dst_col_right, 0, w - 1)
            dst_col_left = int(dst_col)
            dst_col_left = np.clip(dst_col_left, 0, w - 1)

            dst_row_bottom = int(np.ceil(dst_row))
            dst_row_bottom = np.clip(dst_row_bottom, 0, h - 1)
            dst_row_top = int(dst_row)
            dst_row_top = np.clip(dst_row_top, 0, h - 1)

            dst[dst_row_top, dst_col_left] += src[row, col]
            N[dst_row_top, dst_col_left] += 1

            if dst_col_right != dst_col_left:
                dst[dst_row_top, dst_col_right] += src[row, col]
                N[dst_row_top, dst_col_right] += 1

            if dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_left] += src[row, col]
                N[dst_row_bottom, dst_col_left] += 1

            if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_right] += src[row, col]
                N[dst_row_bottom, dst_col_right] += 1

    dst = np.round(dst / (N + 1E-6))
    dst = dst.astype(np.uint8)
    return dst


def backward(src, M, fit=False):
    #####################################################
    # TODO                                              #
    # backward 완성                                      #
    #####################################################
    print('< backward >')
    print('M')
    print(M)
    h_src, w_src = src.shape
    a, b, minCol, minRow = check_size(src, M)

    if not fit:
        dst = np.zeros((h_src, w_src))
    else:

        M[0, 2] -= minCol
        M[1, 2] -= minRow
        dst = np.zeros((a, b))
        h, w = dst.shape
        M_inv = np.linalg.inv(M)

        print('M inv')
        print(M_inv)

        for row in range(h):
            for col in range(w):
                P_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])

                P = np.dot(M_inv, P_dst)
                src_col = P[0, 0]
                src_row = P[1, 0]

                src_col_right = int(np.ceil(src_col))
                src_col_left = int(src_col)

                src_row_bottom = int(np.ceil(src_row))
                src_row_top = int(src_row)

                if src_col_right >= w_src or src_row_bottom >= h_src or src_col_left < 0 or src_row_top < 0:
                    continue

                s = src_col - src_col_left
                s = np.clip(s, 0, 1)
                t = src_row - src_row_top
                t = np.clip(t, 0, 1)

                intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                            + s * (1 - t) * src[src_row_top, src_col_right] \
                            + (1 - s) * t * src[src_row_bottom, src_col_left] \
                            + s * t * src[src_row_bottom, src_col_right]

                dst[row, col] = intensity

        dst = dst.astype(np.uint8)
        return dst

    h, w = dst.shape

    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    for row in range(h):
        for col in range(w):
            P_dst = np.array([
                [col],
                [row],
                [1]
            ])

            P = np.dot(M_inv, P_dst)
            src_col = P[0, 0]
            src_row = P[1, 0]

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)

            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)

            if src_col_right >= w_src or src_row_bottom >= h_src or src_col_left < 0 or src_row_top < 0:
                continue

            s = src_col - src_col_left
            t = src_row - src_row_top

            intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                        + s * (1 - t) * src[src_row_top, src_col_right] \
                        + (1 - s) * t * src[src_row_bottom, src_col_left] \
                        + s * t * src[src_row_bottom, src_col_right]

            dst[row, col] = intensity

    dst = dst.astype(np.uint8)
    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    #####################################################
    # TODO                                              #
    # M 완성                                             #
    # M_tr, M_sc ... 등등 모든 행렬 M 완성하기              #
    #####################################################
    # translation
    M_tr = np.array([
        [1, 0, -30],
        [0, 1, 50],
        [0, 0, 1]
    ])

    # scaling
    M_sc = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1]
    ])

    # rotation
    degree = -20
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])

    # shearing
    M_sh = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])

    # rotation -> translation -> Scale -> Shear
    M = np.dot(M_sh, np.dot(M_sc, np.dot(M_tr, M_ro)))

    # fit이 True인 경우와 False인 경우 다 해야 함.
    fit = True

    # forward
    dst_for = forward(src, M, fit=fit)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    # backward
    dst_back = backward(src, M, fit=fit)
    dst_back2 = backward(dst_back, np.linalg.inv(M), fit=fit)

    cv2.imshow('original', src)
    cv2.imshow('forward2', dst_for2)
    cv2.imshow('backward2', dst_back2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
