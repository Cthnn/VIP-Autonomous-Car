import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import math
print(cv2.__version__)
for file in glob.glob('C:\\Users\\ethan\\Documents\\307\\HW1\\Frames\\*.jpg'):
    print(file)
    img = cv2.imread(file)
    img = cv2.resize(img, (640, 480))
    [r, c, ch] = img.shape
    rStart = int(6*r/7)
    rEnd = int(7*r/7) 
    #print(r, c, ch)


    croppedRoad = img[rStart:rEnd, 0:c]
    hsv = cv2.cvtColor(croppedRoad,cv2.COLOR_BGR2HSV)
    hsv_gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(hsv_gray, 100, 255, 0)
    test,contours, hiearchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv_color = cv2.cvtColor(hsv_gray, cv2.COLOR_GRAY2BGR)
    angles = []
    black_img = np.zeros_like(hsv_color)
    minmax = []
    if(len(contours) < 10):
        for c in range (0,len(contours)):
            vals = {"max_X":-math.inf,"min_X":math.inf,"max_Y":-math.inf,"min_Y":math.inf,"maxY_maxX":-math.inf}
            minmax.append(vals)
    else:
        vals = {"max_X":-math.inf,"min_X":math.inf,"max_Y":-math.inf,"min_Y":math.inf}
        minmax.append(vals)
    for c in range (0,len(contours)):
        if len(contours) < 10:
            #for each contour find its min/max values
            print(len(contours))
        else:
            #this is one contour, therefore iterate and find the min/max
            if contours[c][0][0][0] > minmax[0]['max_X']:
                minmax[0]["max_X"] = contours[c][0][0][0]
            if contours[c][0][0][1] > minmax[0]['max_Y']:
                minmax[0]["max_Y"] = contours[c][0][0][1]
            if contours[c][0][0][0] < minmax[0]['min_X']:
                minmax[0]["min_X"] = contours[c][0][0][0]
            if contours[c][0][0][1] > minmax[0]['min_Y']:
                minmax[0]["min_Y"] = contours[c][0][0][1]
        if cv2.contourArea(contours[c]) > 400:
            contour_img = cv2.drawContours(black_img,contours,c,(255,255,255),3)
    for c in range (0,len(contours)):
        if len(contours) < 10:
            #for each contour find its maxY_maxX
        else:
            #this is one contour, therfore iterate and find the maxY_maxX
            if contours[c][0][0][1] == minmax[0]['max_Y']:
                if contours[c][0][0][0] > minmax[0]['maxY_maxX']:
                    minmax[0]['maxY_maxX'] = contours[c][0][0][0] 
    ret, contour_img = cv2.threshold(contour_img, 100, 255, 0)
    black_img = np.zeros_like(hsv_color)

    height = contour_img.shape[0]
    width = contour_img.shape[1]
    src = np.float32([[0,height],[width,height],[0,0],[width,0]])
    dst = np.float32([[(width/2)-(width*0.45),height],[(width/2)+(width*0.45),height],[0,0],[width,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(contour_img, M, (width, height))
    f,axarr = plt.subplots(1,2,figsize=(30,30))
    axarr[0].imshow(warped_img)
    axarr[1].imshow(cv2.cvtColor(croppedRoad, cv2.COLOR_BGR2RGB))
    axarr[0].set_title('B.E.V Lane Extracted')
    axarr[1].set_title('Original Road Segment')
    plt.show()
    
    # linesP = cv2.HoughLinesP(contour_img, 1, np.pi / 180, 50, None, 50, 0)
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(black_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    # plt.imshow(contour_img)
    # plt.show()

    # cv2.imshow('i',contour_img)
    # cv2.waitKey(0)
'''
img = cv2.imread('track.jpg')
img = cv2.resize(img, (640, 480))
[r, c, ch] = img.shape
rStart = int(r/3)
rEnd = int(2*r/3)
#print(r, c, ch)
croppedRoad = img[rStart:rEnd, 0:c]

hsv = cv2.cvtColor(croppedRoad,cv2.COLOR_BGR2HSV)
hsv_gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(hsv_gray, 100, 255, cv2.THRESH_BINARY) 
#edges = cv2.Canny(gray,150,200)
#lines = cv2.HoughLines(gray,1,np.pi/180, 200)
#print(lines)

for r,theta in lines[0]:
    a = np.cos(theta) 
    b = np.sin(theta) 
    
    x0 = a*r 
    y0 = b*r 
    
    x1 = int(x0 + 1000*(-b))  
    y1 = int(y0 + 1000*(a)) 

    x2 = int(x0 - 1000*(-b))  
    y2 = int(y0 - 1000*(a)) 
    
    cv2.line(croppedRoad,(x1,y1), (x2,y2), (0,0,255),2)
'''
