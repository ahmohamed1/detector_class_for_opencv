import argparse

import cv2.dnn
import numpy as np

class Detector:
    def __init__(self, model_files, class_files, image_size = 640):
        # Load the ONNX model
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_files)
        with open(class_files, 'r') as f:
            image_net_names = f.read().split('\n')
        # final class names (just the first word of the many ImageNet names for one image)
        self.CLASSES = [name.split(',')[0] for name in image_net_names]
        self.colors = [(255,0,0), (255,255,0), (150,0,0), (0,0,255), (0,255,0)]
        self.image_size = image_size #640
    
    def process_image(self, input_image, _confidence=0.7, show=False):
        self.confidence = _confidence
        # Read the input image
        original_image: np.ndarray = input_image.copy()
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / self.image_size

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(self.image_size, self.image_size), swapRB=True)
        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.60, 0.45, 0.5)

        detections = []
        classes_names = []
        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": self.CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            classes_names.append(self.CLASSES[class_ids[index]])
            self.draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        # Display the image with bounding boxes
        if show:
            original_image = cv2.resize(original_image,(940,1024))
            cv2.imshow('image', original_image)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
        return detections, original_image, classes_names
    
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        label = f'{self.CLASSES[class_id]} ({confidence:.2f})'
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)