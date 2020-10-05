import config

from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np


detector = ObjectDetection()
# detector.setModelTypeAsYOLOv3()       # big boi
# detector.setModelTypeAsRetinaNet()    # aam admi
detector.setModelTypeAsTinyYOLOv3()     # smol
detector.setModelPath("/home/pratt3000/Documents/projects/Violence Detection-v3/models/object_detection/yolo-tiny.h5")
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

def mask_frame(frame, detections, detections_temp):
    mask = np.zeros(frame.shape, dtype=np.uint8)

    for eachObject in detections:
        x1,y1,x2,y2 = eachObject["box_points"]
        mask = cv2.rectangle(mask, (x1, y1),(x2,y2), (255,255,255), -1) 

    for eachObject in detections_temp:
        x1,y1,x2,y2 = eachObject["box_points"]
        mask = cv2.rectangle(mask, (x1, y1),(x2,y2), (255,255,255), -1) 

    result = cv2.bitwise_and(frame, mask)   # Mask input image with binary mask
    result[mask==0] = 255   # Optional : set background -> now white/ by default black
    
    return result

def text_to_frame(frame, label):

    cv2.putText(frame, label, (0, 10), config.FONT, fontScale = 0.3, color = (0, 0, 255), thickness=1)

    return frame

def get_frame_difference(frame_1, frame_2):
    frame = cv2.absdiff(frame_1, frame_2)
    ret,thresh1 = cv2.threshold(frame,config.THRESHOLD_DIFF_TOLERANCE,255,cv2.THRESH_BINARY)   #127 is threshold
    return thresh1


vidcap = cv2.VideoCapture(0)
success = True
frame_temp=np.zeros((config.IMG_SIZE,config.IMG_SIZE)).astype(np.uint8) 
stacked_frames=[]
detections_temp = []
count=0

while(success):

    success, frame = vidcap.read()                  # get frame serially
    frame = cv2.resize(frame, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    detections = get_boxes(frame)                   # get obj detection
    
    frame = mask_frame(frame, detections, detections_temp)          # mask is not needed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = get_frame_difference(frame, frame_temp)  # get difference btw frames
    
    # if count < config.FRAME_BATCH_SIZE:
    #     stacked_frames.append(frame_diff)
    #     count += 1
    # else:
    #     stacked_frames=[]
    #     count = 0

    frame_temp = frame
    detections_temp = detections

    model = get_model()
    model.load_weights("/home/pratt3000/Documents/projects/Violence Detection-v3/models/classification/saved-model-01-0.56.hdf5")
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    label = 'Violence'                      #add CNN ka evaluate here
    image = text_to_frame(frame_diff, label)

    image = cv2.resize(image, (1000, 1000))
    cv2.imshow('result', image)
    
    cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break