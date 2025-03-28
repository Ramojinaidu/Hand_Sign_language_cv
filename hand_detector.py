import math

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark


class Hand_detector:
    __BASE_MODEL: str = "models/hand_landmarker.task"

    def __init__(self,
                 num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model: str = __BASE_MODEL):
        self.handedness = None
        self.bbox_data = []
        self.bbox_colour = (0,0,255)
        self.hand_landmark_list = None
        self.landmarks_result = None
        self.num_hands: int = num_hands
        self.min_hand_detection_confidence: float = min_hand_detection_confidence
        self.min_hand_presence_confidence: float = min_hand_presence_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model = model
        self.image = None
        self.draw_landmarks = True
        self.bbox_offset_x = 0
        self.bbox_offset_y = 0

        base_options = python.BaseOptions(self.model)
        options = vision.HandLandmarkerOptions(base_options,
                                               num_hands=self.num_hands,
                                               min_hand_detection_confidence=self.min_hand_detection_confidence,
                                               min_hand_presence_confidence=self.min_hand_presence_confidence,
                                               min_tracking_confidence=self.min_tracking_confidence)
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def image_formater(image: np.ndarray):
        mp_img = mp.Image(mp.ImageFormat.SRGB, cv.cvtColor(image, cv.COLOR_BGR2RGB))
        return mp_img

    def get_landmarks(self, img: np.ndarray):
        mp_img = self.image_formater(img)
        landmarks = self.detector.detect(mp_img)
        return landmarks

    def draw_detection(self, img: np.ndarray = None):
        self.image =img
        self.landmarks_result = self.get_landmarks(self.image)
        self.hand_landmark_list = self.landmarks_result.hand_landmarks
        self.handedness = self.landmarks_result.handedness

        if not self.landmarks_result:
            return self.image



        if self.draw_landmarks:
            for idx in range(len(self.hand_landmark_list)):
                hand_land_marks = self.hand_landmark_list[idx]
                land_marks = landmark_pb2.NormalizedLandmarkList()
                land_marks.landmark.extend([NormalizedLandmark(landmark.x,landmark.y,landmark.z).to_pb2() for landmark in hand_land_marks])
                drawing_utils.draw_landmarks(self.image, land_marks, connections=HAND_CONNECTIONS,
                                             landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 255),
                                                                                             thickness=1,
                                                                                             circle_radius=2),
                                             connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0),
                                                                                               thickness=1))

    def draw_bbox(self, img = None,offset_x = 20,offset_y = 20,draw_box =True):
        if self.image is None : self.image =  img
        self.bbox_offset_x = offset_x
        self.bbox_offset_y = offset_y
        img_height, img_width, _ = self.image.shape# Get the dimensions of the image
        if self.hand_landmark_list:
            for hand_landmark in self.hand_landmark_list:
                # Scale landmark values to pixel values
                hand_landmark_x_values = [int(landmarks.x * img_width) for landmarks in hand_landmark]
                hand_landmark_y_values = [int(landmarks.y * img_height) for landmarks in hand_landmark]

                # Determine the bounding box coordinates
                pts_max = (
                max(hand_landmark_x_values) + self.bbox_offset_x, max(hand_landmark_y_values) + self.bbox_offset_y)
                pts_min = (
                min(hand_landmark_x_values) - self.bbox_offset_x, min(hand_landmark_y_values) - self.bbox_offset_y)
                self.bbox_data = [pts_max, pts_min]

            # Draw the rectangle
            if draw_box:
                cv.rectangle(self.image, pts_min, pts_max, self.bbox_colour, thickness=2, lineType=cv.LINE_8)  # Changed color to red for better visibility

    def get_detected_image(self,flip : bool = True):
        if flip: self.image = cv.flip(self.image, 1)
        return self.image

    def get_cropped_image(self, flip=False, image_size=[300, 300]):
        image_white = np.ones(image_size + [3], dtype=np.uint8) * 255

        # Check if hand landmarks are detected
        if self.hand_landmark_list:
            # Get bounding box information
            try:
                maximum_bbox_point, minimum_bbox_point = self.bbox_data[0], self.bbox_data[1]

                # Ensure bounding box coordinates are within image bounds
                min_x, min_y = max(minimum_bbox_point[0], 0), max(minimum_bbox_point[1], 0)
                max_x, max_y = min(maximum_bbox_point[0], self.image.shape[1]), min(maximum_bbox_point[1], self.image.shape[0])

                # Slice the image based on the bounding box
                img_cropped = self.image[min_y:max_y, min_x:max_x]

                # Check if img_cropped is valid
                if img_cropped.size == 0:
                    # If empty, return blank image
                    return image_white

                # Resize and place on a white background
                h, w = img_cropped.shape[:2]
                if h > w:
                    k = image_size[0] / h
                    wcal = int(k * w)
                    img_cropped = cv.resize(img_cropped, (wcal, image_size[1]))
                    center_offset_x = (image_size[0] - wcal) // 2
                    image_white[:, center_offset_x:center_offset_x + wcal] = img_cropped
                else:
                    k = image_size[1] / w
                    hcal = int(k * h)
                    img_cropped = cv.resize(img_cropped, (image_size[0], hcal))
                    center_offset_y = (image_size[1] - hcal) // 2
                    image_white[center_offset_y:center_offset_y + hcal, :] = img_cropped

            except Exception as e:
               print(f"Error during cropping: {e}")
               return image_white

        # Flip the resulting image if required
        if flip:
            image_white = cv.flip(image_white, 1)

        return image_white

