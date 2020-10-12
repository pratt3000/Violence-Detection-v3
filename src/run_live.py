import config

import os
import cv2
import numpy as np

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
count=0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); 
fgbg = cv2.createBackgroundSubtractorMOG2();  

while(success):

    success, frame = vidcap.read()                  # get frame serially
    frame = cv2.resize(frame, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    frame = fgbg.apply(frame);
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel);
    # if count < config.FRAME_BATCH_SIZE:
    #     stacked_frames.append(frame_diff)
    #     count += 1
    # else:
    #     stacked_frames=[]
    #     count = 0

    label = 'Violence'                      #add CNN ka evaluate here
    frame = text_to_frame(frame, label)

    frame = cv2.resize(frame, (1000, 1000))
    cv2.imshow('result', frame)

    cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break