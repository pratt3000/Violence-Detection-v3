import config

from imageai.Detection import ObjectDetection
import os
import cv2
import random
import numpy as np


# def process_frames(frame, detections_temp):
    
#     frame = cv2.resize(frame, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
#     _, detections = detector.detectCustomObjectsFromImage(
#         custom_objects=custom,
#         input_type="array",
#         input_image= frame,
#         output_type="array"
#         )
    
#     frame = mask_frame(frame, detections, detections_temp)   
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     return frame, detections

# def get_boxes(frame):
#     _, detections = detector.detectCustomObjectsFromImage(
#         custom_objects=custom,
#         input_type="array",
#         input_image= frame,
#         output_type="array"
#         )
#     return detections


def get_frame_difference(frame_1, frame_2):
    frame = cv2.absdiff(frame_1, frame_2)
    ret, thresh1 = cv2.threshold(frame, config.THRESHOLD_DIFF_TOLERANCE, 255, cv2.THRESH_BINARY)   #127 is threshold
    return thresh1

def mask_frame(frame, detections, detections_temp):
    mask = np.zeros(frame.shape, dtype=np.uint8)

    for eachObject in detections:
        x1, y1, x2, y2 = eachObject["box_points"]
        mask = cv2.rectangle(mask, (x1, y1),(x2, y2), (255,255,255), -1) 

    for eachObject in detections_temp:
        x1, y1, x2, y2 = eachObject["box_points"]
        mask = cv2.rectangle(mask, (x1, y1),(x2, y2), (255, 255, 255), -1) 

    result = cv2.bitwise_and(frame, mask)   # Mask input image with binary mask
    result[mask==0] = 255   # Optional : set background -> now white/ by default black

    return result

def get_total_frames(in_dir, dataset_size):
    

    list_fight=os.listdir(os.path.join(in_dir, "Violence"))
    list_no_fight=os.listdir(os.path.join(in_dir, "NonViolence"))
    
    fight_labels = []
    no_fight_labels = []

    for i in range (dataset_size//2):
        fight_labels.append([1, 0])
        no_fight_labels.append([0, 1])

    all_data = list_fight + list_no_fight
    labels = fight_labels + no_fight_labels
    
    total_frames=0  
    
    
    for i in range(dataset_size - (dataset_size%2)): ##no. of videos loop to keep even

        if labels[i]==[1, 0]:
            in_file = os.path.join(in_dir, "Violence")
        else:
            in_file = os.path.join(in_dir, "NonViolence")

        in_file = os.path.join(in_file, all_data[i])
        vidcap = cv2.VideoCapture(in_file)
        
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  #frames in a video\
        total_frames += length
        
    return total_frames


# def my_generater(in_dir, total_videos):     

#     list_fight=os.listdir(os.path.join(in_dir, "Violence"))
#     list_no_fight=os.listdir(os.path.join(in_dir, "NonViolence"))

#     fight_final=random.sample(list_fight, total_videos//2)
#     no_fight_final=random.sample(list_no_fight, total_videos//2)

#     fight_labels = []
#     no_fight_labels = []
#     for i in range (total_videos//2):
#         fight_labels.append([1, 0])
#         no_fight_labels.append([0, 1])

#     final = fight_final + no_fight_final
#     labels = fight_labels + no_fight_labels

#     c = list(zip(final, labels))
#     random.shuffle(c)
#     names, labels = zip(*c)
    
#     images_batches=[]
#     labelss=[]
#     counter = 0
    
#     while True: #select one video per loop

#         video_number = random.randint(1,total_videos-1)

#         if labels[video_number]==[1, 0]:
#             in_file = os.path.join(in_dir, "Violence")
#         else:
#             in_file = os.path.join(in_dir, "NonViolence")

#         in_file = os.path.join(in_file, names[video_number])
#         vidcap = cv2.VideoCapture(in_file)
#         length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  #frames in a video

#         flag_valid_video = 0

#         for j in range(int(length/config.FRAME_BATCH_SIZE)):
            
#             detections_temp_2=[]
#             images_frame_batches=[]
            
#             success,frame_temp = vidcap.read()
#             if success == False:
#                 break

#             frame_temp, detections_temp = process_frames(frame_temp, detections_temp_2)
            
#             for k in range(config.FRAME_BATCH_SIZE - 1):

#                 success, frame = vidcap.read()
#                 if success == False:
#                     flag_valid_video = 1
#                     break
                    
#                 frame, detections = process_frames(frame, detections_temp)                   
                 
#                 diff = get_frame_difference(frame, frame_temp)/255
#                 images_frame_batches.append(diff)

#                 frame_temp = frame
#                 detections_temp=detections

#             if flag_valid_video == 1:
#                 labelss=[]
#                 images_batches=[]
#                 counter = 0
#                 break              

#             if counter < config.BATCH_SIZE:
#                 images_batches.append(images_frame_batches)
#                 counter = counter + 1
#                 labelss.append(labels[video_number])

#             else:
#                 yield np.array(images_batches).reshape((config.BATCH_SIZE, config.FRAME_BATCH_SIZE-1, config.IMG_SIZE, config.IMG_SIZE, 1)), np.array(labelss)
#                 labelss=[]
#                 images_batches=[]
#                 counter = 0


