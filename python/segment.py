import cv2
import numpy as np

img = cv2.imread('./IMG_1153.jpg')
img = cv2.resize(img, (640, 480))

r , c, chr = img.shape
x = int(r/5)
segment_five = img[2*x:3*x,0:c]

cv2.imshow("img",segment_five)
cv2.waitKey(0)