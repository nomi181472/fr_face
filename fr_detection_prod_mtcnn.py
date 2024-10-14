import pandas as pd
from pathlib import Path
from facenet_pytorch import MTCNN
from face_classification import FaceClassification
import torch
import numpy as np
import cv2
from PIL import  Image,ImageOps
import torch.nn.functional as F
from  utility import get_normalize
from utility import get_transformation
from torchvision import models, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transform,test_transform=get_transformation()
resize=transforms.Resize(128)
class FaceDetection:
    def __init__(self,face_detector_input_image_size):

        self.face_detector_input_image_size=face_detector_input_image_size
        self.load_pretrained_weight()



    def load_pretrained_weight(self):
        # If required, create a face detection pipeline using MTCNN:
        self.mtcnn = MTCNN(  device=device)

    @staticmethod
    def get_resize(resize_width, frame):
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        new_height = int(resize_width * aspect_ratio)
        frame_resized = cv2.resize(frame, (resize_width, new_height))
        return frame_resized

    def face_recognize(self, video_path,   prob_threshold = 0.7) -> None:

        cam = cv2.VideoCapture(str(video_path))
        cv2.namedWindow("Capture Face")

        while True:
            try:
                ret, frame = cam.read()
                if not ret:
                    break
                # mtcnn
                boxes, probs = self.mtcnn.detect(frame)
                if boxes is not None and probs is not None:
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        if prob >= prob_threshold:
                            x1, y1, x2, y2, face = self.get_box_and_faces(frame, box)
                            if face.size == 0:
                                print("zero size face")
                                continue

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            print(f"face prob mismatch:{prob} with {prob_threshold}")
                else:
                    print('box or probs not found')
                cv2.imshow("Capture Face", frame)

                k = cv2.waitKey(1)
                if k % 256 == 27:  # ESC or 20 images collected
                    break
            except Exception as e:
                print(e)
                print("exception")
                break

        cam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_box_and_faces(frame_resized, box):
        x1, y1, x2, y2 = map(int, box)  # Convert to int
        # Draw rectangle around the detected face

        face = frame_resized[y1:y2, x1:x2]
        return x1, y1, x2, y2, face




obj = FaceDetection(
face_detector_input_image_size=460
)
obj.face_recognize("rtsp://admin:DHA@1431@192.168.80.250:554/Streaming/Channels/101",)

