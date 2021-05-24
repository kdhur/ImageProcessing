def my_normalize(src):
    dst = src.copy()
    if dst.max() == dst.min():
        dst *= 255
        return dst
    dst = ((dst - dst.min()) * 255 / (dst.max() - dst.min()))
    return dst
