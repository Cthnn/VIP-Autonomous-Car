import cv2

img = cv2.imread("./right_turn.jpg")
new_img = cv2.flip(img,1)
cv2.imwrite("./left_turn.jpg", new_img)
