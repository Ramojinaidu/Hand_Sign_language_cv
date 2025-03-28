import hand_detector
import image_classifier
import cv2 as cv
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal,QSize,Qt
import sys
from PyQt5.QtGui import QImage,QPixmap
import numpy as np
import time
import h5py

from image_classifier import Classifier


class app_gui(QMainWindow):
    def __init__(self):
        super(app_gui,self).__init__()
        uic.loadUi("./user_interface/app.ui",self)
        self.output_string = ""

        self.text_area : QTextEdit = self.findChild(QTextEdit,"text_panel")
        self.clear_btn : QPushButton = self.findChild(QPushButton,"clear_button")
        self.video_screen : QLabel = self.findChild(QLabel,"video_panel")
        self.clear_btn.clicked.connect(self.clear_text)
        self.video_feed = Video_feed()
        self.video_feed.start()
        self.video_feed.image_update.connect(self.update_video_screen)
        self.video_feed.char_predicted.connect(self.update_output_string)


    def video_stop(self):
        self.video_feed.stop()

    def clear_text(self):
        self.output_string = ""
        self.text_area.setText("")


    def update_video_screen(self,image):
        self.video_screen.setPixmap(QPixmap.fromImage(image).scaled(QSize(self.video_screen.width(),self.video_screen.height())))


    def update_output_string(self,char):
        if len(self.output_string)!=0 and char=="del":
            self.output_string = self.output_string[:-1]
        elif char == "spc":
            self.output_string += " "
        else:
            self.output_string += char

        self.text_area.setText(self.output_string)





        self.show()


class Video_feed(QThread):
    image_update = pyqtSignal(QImage)
    char_predicted = pyqtSignal(str)
    def __init__(self):
        super().__init__()

        self.thread_active = True
        self.frame = cv.VideoCapture(0)
        self.detector = hand_detector.Hand_detector()

        f = h5py.File("./models/keras_model1.h5", mode="r+")
        model_config_string = f.attrs.get("model_config")
        if model_config_string.find('"groups": 1,') != -1:
            model_config_string = model_config_string.replace('"groups": 1,', '')
            f.attrs.modify('model_config', model_config_string)
            f.flush()
            model_config_string = f.attrs.get("model_config")
            assert model_config_string.find('"groups": 1,') == -1

        f.close()


        self.classifier = Classifier("./models/keras_model1.h5","./lables/hand_labels.txt")
        self.start_time=time.time()
        self.delay = 4
        self.labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","spc","del"]

    def run(self,):
        while self.thread_active:
            ret, img = self.frame.read()

            self.detector.draw_detection(img)
            self.detector.draw_bbox(draw_box=True)
            image = np.array(self.detector.get_detected_image())
            if ((time.time()-self.start_time) >= self.delay) and self.detector.hand_landmark_list:
                p,index = self.classifier.getPrediction(cv.cvtColor(self.detector.get_cropped_image(),cv.COLOR_RGB2BGR))
                print(np.argmax(p))
                self.char_predicted.emit(self.labels[index-1])
                self.start_time = time.time()


            qformat_image = QImage(image.data,image.shape[1],image.shape[0],QImage.Format_BGR888)
            self.image_update.emit(qformat_image)

    def stop(self):
        self.thread_active = False
        self.frame.release()
        self.quit()

app = QApplication(sys.argv)
win = app_gui()
app.exec_()