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

    cv2.putText(frame, label, (50, 10), config.FONT, fontScale = 0.1, color = (0, 0, 255), thickness=1)

    return frame

def frame_preprocessing(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    return frame

def get_frame_difference(frame_1, frame_2):
    frame = cv2.absdiff(frame_1, frame_2)
    print(frame.shape)
    ret,thresh1 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)   #127 is threshold
    return thresh1


vidcap = cv2.VideoCapture(0)
success = True
frame_temp=np.zeros((config.IMG_SIZE,config.IMG_SIZE)).astype(np.uint8) 
stacked_frames=[]
count=0
while(success):

    success, frame = vidcap.read()
    
    detections = get_boxes(frame)           #get obj detection
    frame = mask_frame(frame, detections)   #mask
    frame = frame_preprocessing(frame)      #grayscale     
    

    # diff = get_frame_difference(frame, frame_temp)
    # if count < config.FRAME_BATCH_SIZE:
    #     stacked_frames.append(diff)
    #     count += 1
    # else:
    #     stacked_frames=[]
    #     count = 0

    # frame_temp = frame

    label = 'Violence'                      #add CNN ka evaluate here
    frame = text_to_frame(frame, label)

    cv2.imshow('result', frame)
    cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

