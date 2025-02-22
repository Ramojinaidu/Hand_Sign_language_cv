import cv2 as cv

import hand_detector

frame = cv.VideoCapture(0)


detector = hand_detector.Hand_detector()
while True:
    ret, img = frame.read()
    #print(img.dtype)
    k = cv.waitKey(1)
    if k == ord("q"):
        break


    img_l = detector.draw_detection(img,True,True)
    img_f= cv.flip(img_l, 1)

    cv.imshow("video",img_f)
   
frame.release()
cv.destroyAllWindows()
