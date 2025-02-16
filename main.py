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

    img_f= cv.flip(img, 1)
    img_l = detector.draw_landmarks(img_f)
    cv.imshow("video",img_l)
   
frame.release()
cv.destroyAllWindows()
