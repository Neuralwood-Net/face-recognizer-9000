import cv2
import numpy as np
from mtcnn import MTCNN

from image_processing import crop_images, draw_class_names, get_bounding_boxes
from recognizer.recognizer import Recognizer


class Demo:
    def __init__(self):
        self.detector = MTCNN()
        self.cap = cv2.VideoCapture(0)
        self.recognizer = Recognizer()
        self.labels = ["Kjartan", "Lars", "Morgan", "Other"]
        self.allow_duplicate_predictions = True

    def run_demo(self):
        print("Starting demo")

        while True:
            # Capture frame-by-frame
            _, frame = self.cap.read()

            # Detect faces
            bounding_boxes = get_bounding_boxes(frame, self.detector)

            # Crop to square around bounding boxes and draw bounding boxes on display image
            cropped, display_image = crop_images(frame, bounding_boxes)

            # Resize cropped images to input size of neural net
            resized = [cv2.resize(im, (224, 224)) / 255.0 for im in cropped]

            # Predict classes for faces
            pred_probs, pred_labels = self.recognizer.recognize(
                np.array(resized),
                self.labels,
                allow_duplicates=self.allow_duplicate_predictions,
            )

            # Draw class names and certainties on display image
            draw_class_names(display_image, bounding_boxes, pred_probs, pred_labels)

            # Display the resulting frame
            cv2.imshow("frame", display_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()


demo = Demo()
demo.run_demo()
