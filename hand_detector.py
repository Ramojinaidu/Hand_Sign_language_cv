import cv2 as cv
import mediapipe as mp
import numpy
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.python import solutions



class Hand_detector:
    __BASE_MODEL: str = "model/hand_landmarker.task"

    def __init__(self,
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model: str = __BASE_MODEL):
        self.hand_landmark_list = None
        self.landmarks_result = None
        self.num_hands: int = num_hands
        self.min_hand_detection_confidence: float = min_hand_detection_confidence
        self.min_hand_presence_confidence: float = min_hand_presence_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model = model
        self.image = None

        base_options = python.BaseOptions(self.model)
        options = vision.HandLandmarkerOptions(base_options,
                                               num_hands=self.num_hands,
                                               min_hand_detection_confidence=self.min_hand_detection_confidence,
                                               min_hand_presence_confidence=self.min_hand_presence_confidence,
                                               min_tracking_confidence=self.min_tracking_confidence)
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def image_formater(image: numpy.ndarray):
        mp_img = mp.Image(mp.ImageFormat.SRGB, cv.cvtColor(image, cv.COLOR_BGR2RGB))
        return mp_img

    def get_landmarks(self, img: numpy.ndarray):
        mp_img = self.image_formater(img)
        landmarks = self.detector.detect(mp_img)
        return landmarks

    def draw_detection(self, img: numpy.ndarray = None,draw_landmarks:bool=True):
        self.image =img
        self.landmarks_result = self.get_landmarks(self.image)
        self.hand_landmark_list = self.landmarks_result.hand_landmarks
        handedness = self.landmarks_result.handedness
        print(handedness)

        if not self.landmarks_result:
            return self.image



        if draw_landmarks:
            for idx in range(len(self.hand_landmark_list)):
                hand_land_marks = self.hand_landmark_list[idx]
                land_marks = landmark_pb2.NormalizedLandmarkList()
                land_marks.landmark.extend([NormalizedLandmark(landmark.x,landmark.y,landmark.z).to_pb2() for landmark in hand_land_marks])
                drawing_utils.draw_landmarks(self.image,land_marks,solutions.hands.HAND_CONNECTIONS)

    def draw_bbox(self, img = None,offset_x = 20,offset_y = 20):
        if self.image is None : self.image =  img
        img_height, img_width, _ = self.image.shape# Get the dimensions of the image
        for hand_landmark in self.hand_landmark_list:
            # Scale landmark values to pixel values
            hand_landmark_x_values = [int(landmarks.x * img_width) for landmarks in hand_landmark]
            hand_landmark_y_values = [int(landmarks.y * img_height) for landmarks in hand_landmark]

            # Determine the bounding box coordinates
            pts_max = (max(hand_landmark_x_values)+offset_x, max(hand_landmark_y_values)+offset_y)
            pts_min = (min(hand_landmark_x_values)-offset_x, min(hand_landmark_y_values)-offset_y)

            # Draw the rectangle
            cv.rectangle(self.image, pts_min, pts_max, (0, 0, 255), thickness=2,
                         lineType=cv.LINE_8)  # Changed color to red for better visibility

    def get_detected_image(self,flip : bool = True):
        if flip: self.image = cv.flip(self.image, 1)

        return self.image
