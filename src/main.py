import config

from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(config.PRETRAINED_MODEL_PATH)
detector.loadModel(detection_speed = config.OBJECT_DETECTION_SPEED) #change parameter to adjust accuracy and speed
custom = detector.CustomObjects(person=True)


def get_boxes(frame):
    _, detections = detector.detectCustomObjectsFromImage(
        custom_objects=custom,
        input_type="array",
        input_image= frame,
        output_type="array"
        )
    return detections

def mask_frame(frame, detections):
    mask = np.zeros(frame.shape, dtype=np.uint8)

    for eachObject in detections:
        x1,y1,x2,y2 = eachObject["box_points"]
        mask = cv2.rectangle(mask, (x1, y1),(x2,y2), (255,255,255), -1) 

    result = cv2.bitwise_and(frame, mask)   # Mask input image with binary mask
    result[mask==0] = 255   # Optional : set background -> now white/ by default black
    
    return result

def text_to_frame(frame, label):
    font = config.FONT
    cv2.putText(frame, label, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)

    return frame


vidcap = cv2.VideoCapture(0)
success = True
while(success):

    success, frame = vidcap.read()

    detections = get_boxes(frame)
    masked_frame = mask_frame(frame, detections)
    label = 'Violence'  #add CNN ka jo bhi here 
    frame_with_label = text_to_frame(masked_frame, label)

    cv2.imshow('result', frame_with_label)
    cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

