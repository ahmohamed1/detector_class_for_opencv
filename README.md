# Introduction
This repository provides a Python class for integrating the YOLOv8 object detection model, converted to ONNX format, with OpenCV. This setup allows you to leverage the trained model.onnx along with the labels.txt file that accompanies your training data.

# How to use

To use the detector, include the following header in your Python script:

```python
From detector import Detector
```
Create an instance of the Detector class by specifying the path to your ONNX model and the labels file. You also need to set the image_size parameter appropriate for your model:
```python
detector = Detector("model.onnx", 'labels.txt',image_size=)
```
In your main processing loop, use the process_image method to detect objects within frames captured from a video source:
```python
_, img, classes_names = detector.process_image(frame)
```

## Complete Sample Code

Below is a complete sample script that uses the detector with a webcam feed:

```python
import cv2
detector = Detector("best.onnx", 'labels.txt',image_size=410)
cap = cv2.VideoCapture(0)

while True:
    # capture frame
    ret, frame = cap.read()
    if ret == True:
        _, img, classes_names = detector.process_image(frame)
        # Display the resulting frame with detections
        ikey = cv2.imshow('Frame',img)
         # Press 'q' to exit the loop
        if ikey == ord('q'):
            break

cap.release()  
cv2.destroyAllWindows()
```

