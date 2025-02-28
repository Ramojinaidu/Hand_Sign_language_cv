import cv2 as cv

import hand_detector

frame = cv.VideoCapture(0)

detector = hand_detector.Hand_detector()
while True:
    ret, img = frame.read()
    k = cv.waitKey(1)
    if k == ord("q"):
        break

    detector.draw_detection(img)
    detector.draw_bbox()

    cv.imshow("video", detector.get_detected_image())

frame.release()
cv.destroyAllWindows()
