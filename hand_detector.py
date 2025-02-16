import cv2 as cv
import mediapipe as mp
import numpy
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
class Hand_detector:
    __BASE_MODEL: str = "model/hand_landmarker.task"

    def __init__(self,
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model: str = __BASE_MODEL):
        self.num_hands: int = num_hands
        self.min_hand_detection_confidence: float = min_hand_detection_confidence
        self.min_hand_presence_confidence: float = min_hand_presence_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model = model

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

    def draw_landmarks(self, img: numpy.ndarray):
        landmarks_result = self.get_landmarks(img)

        # Ensure that landmarks exist before processing
        if not landmarks_result.hand_landmarks:
            return img

        # Draw landmarks on the image
        h, w, _ = img.shape
        for landmark in landmarks_result.hand_landmarks[0]:
            x, y = int(landmark.x * w), int(landmark.y * h)
            img = cv.circle(img, (x, y), 2, (0, 0, 0), cv.LINE_8)

        return img
