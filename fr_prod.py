import pandas as pd
from pathlib import Path
from facenet_pytorch import MTCNN
from face_classification import FaceClassification
import torch
import numpy as np
import cv2
from PIL import  Image
import torch.nn.functional as f
from utility import get_transformation
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transform,test_transform=get_transformation()
resize=transforms.Resize(128)

class FRBaseFacenet:
    def __init__(self, classifier_weight_path, id_name_csv_path, classifier_input=128):
        names_df = pd.read_csv(id_name_csv_path, index_col=0)
        self.ids_name = names_df[["name", "id"]].groupby(["name", "id"]).mean().reset_index()
        self.classifier_weight_path = Path(classifier_weight_path)
        self.classifier_input = classifier_input
        print(names_df.to_dict(orient='list'))
        self.mtcnn=None
        self.classifier=None
        self.load_pretrained_weight()

    def get_class_name(self, pred_id):
        name = self.ids_name[self.ids_name["id"] == pred_id]["name"].values[-1]
        return name.split("_")[-1]

    def load_pretrained_weight(self):
        # If required, create a face detection pipeline using MTCNN:
        self.mtcnn = MTCNN(image_size=self.classifier_input, device=device)
        try:
            self.classifier = FaceClassification(len(self.ids_name))
            self.classifier.load_state_dict(torch.load(str(self.classifier_weight_path)))
            print(f"loading model in {device}")
            if device == 'cuda':
                self.classifier = self.classifier.cuda()
            print(f"classifier weight loaded successfully")
        except  Exception as e:
            print(f"classifier loading weight exception occur {e}")



    def face_recognize(self, video_path,prob_threshold = 0.3) -> None:

        cam = cv2.VideoCapture(str(video_path))
        cv2.namedWindow("Capture Face")
        self.classifier.eval()
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

                            self.make_rectangle(frame, x1, x2, y1, y2)
                            if print(face.shape[0]>100 or face.shape[1]>100):
                                resized_img = self.transform_input(face)


                                logits = self.classifier(resized_img)
                                indices, probs = self.get_probability_distribution(logits, probs)
                                self.get_recognized_class(frame, indices, probs, x1, y1)
                            else:
                                print(f"very small image_shape :{face.shape}")
                                cls_name = "small size"
                                text_color = (0, 0, 255)
                                self.put_label(cls_name, frame, text_color, x1, y1)


                        else:
                            print(f"prob_mismatch:given:{prob_threshold}, recieved={prob}")



                else:
                    print("0 boxes")

                cv2.imshow("Capture Face", frame)

                k = cv2.waitKey(1)
                if k % 256 == 27:  # ESC or 20 images collected
                    break
            except Exception as e:
                print(e)
                print("exception")
                break

        self.dispose(cam)

    def get_probability_distribution(self, logits, probs):
        probs = f.softmax(logits, dim=1)
        # print(probs)
        _, indices = torch.max(probs, dim=1)
        return indices, probs

    def get_recognized_class(self, frame, indices, probs, x1, y1):
        for i, at_prob in enumerate(probs):
            index = indices[i].item()
            prob = at_prob[index]
            #             #print(f"prob:{prob}")
            if prob > 0.4:
                text_color = (0, 255, 0)
                cls_name = self.get_class_name(index)
                self.put_label(cls_name, frame, text_color, x1, y1)
            elif prob < 0.2:
                cls_name = "not in database"
                text_color = (0, 0, 255)
                self.put_label(cls_name, frame, text_color, x1, y1)
            else:
                cls_name = "recognizing..."
                text_color = (255, 255, 255)
                self.put_label(cls_name, frame, text_color, x1, y1)
                print(f"cls_name:{cls_name} with prob{prob}")

    @staticmethod
    def dispose(cam):
        cam.release()
        cv2.destroyAllWindows()

    @staticmethod
    def put_label(cls_name, frame_resized, text_color, x1, y1):
        cv2.putText(frame_resized, cls_name, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1,
                    cv2.LINE_AA)  # White text, thicker, anti-aliased

    def transform_input(self, face):

        face_pil = Image.fromarray(face)
        resized_img = face_pil.resize((self.classifier_input, self.classifier_input))
        if resized_img.mode != 'RGB':
            resized_img = resized_img.convert('RGB')
            print("converting mode")
        resized_img = torch.tensor(np.array(resized_img), dtype=torch.float32, device=device)
        bgr_image = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        # Save the image
        output_path = 'after.png'  # Specify the output file name
        cv2.imwrite( output_path,bgr_image)
        resized_img = resized_img.permute(2, 0, 1)
        resized_img = resized_img.unsqueeze(0)
        return resized_img

    @staticmethod
    def make_rectangle(frame_resized, x1, x2, y1, y2):
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def get_box_and_faces(frame_resized, box):
        x1, y1, x2, y2 = map(int, box)  # Convert to int
        # Draw rectangle around the detected face

        face = frame_resized[y1:y2, x1:x2]
        return x1, y1, x2, y2, face

    @staticmethod
    def get_resize(resize_width, frame):
        height, width = frame.shape[:2]
        aspect_ratio = height / width
        new_height = int(resize_width * aspect_ratio)
        frame_resized = cv2.resize(frame, (resize_width, new_height))
        return frame_resized


obj = FRBaseFacenet(

    "./best_model.pth",
    "./name_id.csv"
)
obj.face_recognize("rtsp://admin:DHA@1431@192.168.80.250:554/Streaming/Channels/101",
                   prob_threshold=0.7

                   )

