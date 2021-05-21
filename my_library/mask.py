import numpy as np
import math


def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-msize // 2.0 + 1: msize // 2.0 + 1, -msize // 2.0 + 1: msize // 2.0 + 1]
    gaus2D = x ** 2 + y ** 2

    for i in range(msize):
        for j in range(msize):
            gaus2D[i][j] = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(gaus2D[i][j] / (2 * (sigma ** 2))))

    gaus2D /= np.sum(gaus2D)

    return gaus2D

def get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    gaus2D = 1 / (2 * np.pi * sigma**2) * np.exp(-((x**2 + y**2)/(2 * sigma**2)))
    gaus2D /= np.sum(gaus2D)

    return gaus2D

def get_my_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-fsize // 2.0 + 1: fsize // 2.0 + 1, -fsize // 2.0 + 1: fsize // 2.0 + 1]
    DoG_x = x ** 2 + y ** 2
    DoG_y = x ** 2 + y ** 2

    for i in range(fsize):
        for j in range(fsize):
            DoG_x[i][j] = (-x[i][j] / (sigma ** 2)) * math.exp(-(DoG_x[i][j] / (2 * (sigma ** 2))))
    for i in range(fsize):
        for j in range(fsize):
            DoG_y[i][j] = (-y[i][j] / (sigma ** 2)) * math.exp(-(DoG_y[i][j] / (2 * (sigma ** 2))))

    return DoG_x, DoG_y


def get_DoG_filter(fsize, sigma=1):  # TA_solution
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-(fsize // 2): (fsize // 2) + 1, -(fsize // 2): (fsize // 2) + 1]
    DoG_x = (-x / sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    DoG_y = (-y / sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    #필터 합 0
    # DoG_x = DoG_x - (DoG_x.sum()/fsize**2)
    # DoG_y = DoG_y - (DoG_y.sum()/fsize**2)

    return DoG_x, DoG_y