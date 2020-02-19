import cv2

cap = cv2.VideoCapture(1)
counter = 0
picture_counter = 0
while True:
    ret, img = cap.read()

    cv2.imshow("img", img)

    keyPressed = cv2.waitKey(1)

    if keyPressed == ord('q'):
        cv2.destroyAllWindows()
        break

    if(counter%5 == 0):
        cv2.imwrite(f"center{picture_counter}.jpg",img)
        picture_counter+=1
        counter = 0

    print(counter)
    counter+=1
