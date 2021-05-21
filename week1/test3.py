import cv2
import numpy as np

src = np.zeros((300, 300, 3), dtype=np.uint8)
src[0, 0] = [1, 2, 3] #0,0 픽셀에 BGR 값을 넣어준것임
src[0, 1] = [4, 5, 6]
src[1, 0] = [7, 8, 9]
src[0, 2] = [255, 0, 0]
src[1, 1] = [255, 255, 255]

print(src.shape)
print(src[0,0,0], src[0,0,1], src[0,0,2]) #src 0,0의 B값, G값, R값 출력
print(src[0,0]) #src 0,0의 BGR값 출력
print(src[0]) #src 0row의 BGR값 출력
print(src) #전체 출력

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()