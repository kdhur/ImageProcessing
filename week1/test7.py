import  cv2
import numpy as np

src = cv2.imread('Lena.png')
(h, w, c) = src.shape
yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV) #cv2로 BGR을 휘도(Y)로 변환
my_y = np.zeros((h, w)) #검은색 기본 이미지 my_y로 생성
my_y = (src[:,:,0] * 0.114) + (src[:,:,1] * 0.587) + (src[:,:,2] * 0.299) #직접 BGR을 휘도(Y)로 변환 (gray랑 다름)
my_y = (my_y + 0.5).astype(np.uint8) #== my_y = np.round(my_y).astype(np.uint8) => 반올림

cv2.imshow('original', src)
cv2.imshow('cvtColor', yuv[:,:,0])
cv2.imshow('my_y', my_y)

print(yuv[0:5, 0:5, 0])
print(my_y[0:5, 0:5])

cv2.waitKey()
cv2.destroyAllWindows()