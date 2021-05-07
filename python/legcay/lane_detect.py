import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask(img):
    height = img.shape[0]
    square = np.array([[(75,height),(100,height),(75,150),(100,150)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,square,255)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image
def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0),10)
    return line_image
img = cv2.imread('./right_turn.jpg')

lane_img = cv2.Canny(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),50,150)

masked = mask(lane_img)
cv2.imwrite("./left_turn.jpg", masked)
#lines = cv2.HoughLinesP(masked,2,np.pi/180,100,np.array([]),minLineLength=1,maxLineGap=500)
#line_image =display_lines(lane_img,lines)
while True:
    cv2.imshow("result",lane_img)  
    keyPressed = cv2.waitKey(1)
    if(keyPressed == ord('q')):
        cv2.destroyAllWindows()
        break;
