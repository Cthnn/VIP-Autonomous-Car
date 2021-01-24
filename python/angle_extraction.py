import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math
def euclidean_dist(pt,fixed):
    a = fixed[0]-pt[0]
    b = fixed[1]-pt[1]
    if b == 0:
        return abs(a)
    else:
        dist = math.sqrt((a*a)+(b*b))
        return dist
def corner_search(img):
    corners = {'top_left':(math.inf,math.inf),'top_right':(math.inf,math.inf),'bottom_left':(math.inf,math.inf),'bottom_right':(math.inf,math.inf)}
    found = False
    h = img.shape[0]
    w = img.shape[1]
    for c in range(0,w):
        for r in range(0,h):
            if(img[r][c] > 0):
                if(euclidean_dist((c,r),(0,0)) < euclidean_dist(corners['top_left'],(0,0))):
                    corners['top_left'] = (c,r)
                if(euclidean_dist((c,r),(w-1,0)) < euclidean_dist(corners['top_right'],(w-1,0))):
                    corners['top_right'] = (c,r)
                if(euclidean_dist((c,r),(w-1,h-1)) < euclidean_dist(corners['bottom_right'],(w-1,h-1))):
                    corners['bottom_right'] = (c,r)
                if(euclidean_dist((c,r),(0,h-1)) < euclidean_dist(corners['bottom_left'],(0,h-1))):
                    corners['bottom_left'] = (c,r)
    return corners
def noise_reduct (img):
    # Converting colorspace to remove noise
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(hsv_gray, 120, 255, 0)
    return thresh
def lane_seg(img, contour_thresh = 1000):
    test,contours, hiearchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_img = np.zeros_like(img)
    # Draw the Contours
    all_corners = []
    count = 0
    for c in range (0,len(contours)):
        if cv2.contourArea(contours[c]) > contour_thresh:
            single_contour = np.zeros_like(black_img)
            epsilon = 0.02*cv2.arcLength(contours[c],True)
            approx = cv2.approxPolyDP(contours[c],epsilon,True)
            black_img = cv2.drawContours(black_img,[approx],-1,(255,255,255),3)
            single_contour = cv2.drawContours(single_contour,[approx],-1,(255,255,255),3)
            corners = corner_search(single_contour)
            all_corners.append(corners)
    return black_img,all_corners
def bev(img,src,dst,w,h):
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (w, h))
    return warped_img
def roi(img,vertices):
    #Find Region of Interest
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    return cv2.bitwise_and(img,mask)


def calculate_img(img_param):
    orig_img = img_param
    img = cv2.resize(orig_img, (640, 480))
    # Gaussian Blur to smooth lines and help with edge detection
    img = noise_reduct(img)

    height = img.shape[0]
    width = img.shape[1]
    #HighLight Region Of Interest
    trap_bl = (210,height)
    trap_tl = (273,313)
    trap_tr = (429,313)
    trap_br = (width-29,height)
    vertices = np.array([[trap_bl,trap_tl,trap_tr,trap_br]])
    lane_img = roi(img,vertices)

    #Birds Eye View
    src = np.float32([list(trap_tl),list(trap_tr),list(trap_br),list(trap_bl)])
    out_width = 200
    out_height = 500
    dst = np.float32([[0,0],[out_width,0],[out_width,out_height],[0,out_height]])
    warped = cv2.GaussianBlur(bev(lane_img,src,dst,200,500),(7,7),11)
    #Crop to keep the road close to the car
    crop_start = int(out_height/2)
    warped = warped[crop_start: , 0:]
    #Find Contours (Blurred so that lines are straight)
    res, all_corners = lane_seg(cv2.blur(warped,(11,11)))
    angles = []
    if(all_corners != [] and len(all_corners) < 3):
        for i in range(0,len(all_corners)):
            angle = 0
            tl = (all_corners[i]['top_left'][0],-(all_corners[i]['top_left'][1]-(crop_start-1)))
            tr = (all_corners[i]['top_right'][0],-(all_corners[i]['top_right'][1]-(crop_start-1)))
            bl = (all_corners[i]['bottom_left'][0],-(all_corners[i]['bottom_left'][1]-(crop_start-1)))
            br = (all_corners[i]['bottom_right'][0],-(all_corners[i]['bottom_right'][1]-(crop_start-1)))
            if(abs(tl[0]-bl[0])< 30 and abs(tr[0]-br[0])< 30 and abs(bl[1]-br[1])< 30 and abs(tl[1]-tr[1])< 30):
                continue
            else:
                angle1 = 0
                print(tl,tr,bl,br)
                if(tr[0]-bl[0] != 0):
                    angle1 = 90-math.degrees(math.atan((tr[1]-bl[1])/(tr[0]-bl[0])))
                angles.append(angle1)

                #This code calculates the absolute angle of the intersection of the top line with the side line
                # if(tl[0] - bl[0] < -3):
                #     #left turn
                #     if(br[0]-tl[0] != 0):
                #         slope = abs((br[1]-tl[1])/br[0]-tl[0])
                #         if(slope > 0):
                #             s1 = (tl[1]-tr[1])/(tl[0]-tr[0]) if tl[0]-tr[0] != 0 else 0
                #             s2 = (tr[1]-br[1])/(tr[0]-br[0]) if tr[0]-br[0] != 0 else 0
                #             if(s1 < 0  and s2 == 0):
                #                 angle = 90+math.degrees(math.atan(s1))
                #             elif(s1 == 0 and s2 == 0):
                #                 angle = 90
                #             elif(s2 < 0 and s1 > 0):
                #                 t = (s1-s2)/(1+(s1*s2))
                #                 angle = 180 + math.degrees(math.atan(t))
                #             else:
                #                 t = (s1-s2)/(1+(s1*s2))
                #                 angle = abs(math.degrees(math.atan(t)))
                #     angles.append(tuple(["left",angle]))
                # else:
                    # if(bl[0]-tr[0] != 0):
                    #     slope = abs((bl[1]-tr[1])/(bl[0]-tr[0]))
                    #     if(slope > 0):
                    #         s1 = (tr[1]-tl[1])/(tr[0]-tl[0]) if tr[0]-tl[0] != 0 else 0
                    #         s2 = (tl[1]-bl[1])/(tl[0]-bl[0]) if tl[0]-bl[0] != 0 else 0
                    #         if((s1 > 0  and s2 == 0) or (s2 > 0 and s1 == 0)):
                    #             s1 = max(s1,s2)
                    #             angle = 90+math.degrees(math.atan(s1))
                    #         elif(s1 == 0 and s2 == 0):
                    #             angle = 90
                    #         elif((s1 > 0 and s2) > 0 or (s1 < 0 and s2 < 0)):
                    #             t = (s1-s2)/(1+(s1*s2))
                    #             angle = 180-abs((math.degrees(math.atan(t))))
                    #         else:
                    #             t = (s1-s2)/(1+(s1*s2))
                    #             angle = 90-abs(math.degrees(math.atan(t)))
                    # angles.append(tuple(["right",angle]))
    return angles,res


for i in range(0,260):
    file = 'C:\\Users\\ethan\\Documents\\307\\HW1\\Frames\\frame'+str(i)+'.jpg'
    img = cv2.imread(file)
    angles,res = calculate_img(img)
    f,axarr = plt.subplots(1,2)
    axarr[0].imshow(img)
    axarr[1].imshow(res)
    axarr[0].set_title('Original Road Segment')
    axarr[1].set_title(str(angles))
    plt.show()