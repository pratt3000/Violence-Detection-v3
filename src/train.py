import config

from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np
from PIL import Image


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

def get_frame_difference(stacked_frames):
    stacked_difference_frames=[]
    for i in range(len(stacked_frames)-1):
        
        px1 = stacked_frames[i]
        px2 = stacked_frames[i+1]
        imOut = Image.new('RGB', im1.size)
        pxOut = imOut.load()
        for x in range(0, im1.width):
            for y in range(0, im1.height):
                if px1[x, y] == px2[x, y]:
                    #r, g, b = px1[x, y]
                    #grayscale = int((r + g + b) / 3)
                    #pxOut[x, y] = (grayscale, grayscale, grayscale)
                    pxOut[x, y] = (0, 0, 0)
                else:
                    pxOut[x, y] = (255, 0, 0)
        stacked_difference_frames.append(pxOut)
        i=i+1
    return stacked_difference_frames
    
def stack_frames_until(stacked_frames,frame):
    if len(stacked_frames) < config.NO_OF_FRAMES_TO_STACK:
        stacked_frames.append(frame)
    return stacked_frames
    

vidcap = cv2.VideoCapture(0)
success = True

while(success):
    stacked_frames=[]


    success, frame = vidcap.read()

    detections = get_boxes(frame)
    masked_frame = mask_frame(frame, detections)
    stacked_frames = stack_frames_until(stacked_frames,masked_frame)
    

    #need to change dis line
    stacked_difference_frames=get_frame_difference(stacked_frames)




    
    #cv2.imshow('result', masked_frame)
    #cv2.waitKey(1)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break



    