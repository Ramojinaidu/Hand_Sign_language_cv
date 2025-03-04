import time
import cv2 as cv
import hand_detector
count = 0
frame = cv.VideoCapture(0)
path = "./image_data/B/"
detector = hand_detector.Hand_detector()
while True:
    ret, img = frame.read()
    k = cv.waitKey(1)
    c = cv.waitKey(1)
    if k == ord("q"):
        break

    detector.draw_detection(img)
    detector.draw_bbox(offset_x=18,offset_y=10,draw_box=False)

    cv.imshow("video", detector.get_cropped_image(flip=True))
    if k == ord("c") and detector.handedness:
        count+=1
        cv.imwrite(f"image_data/Y/img_{time.time()}.png",detector.get_cropped_image(True))
        print(count)

frame.release()
cv.destroyAllWindows()
