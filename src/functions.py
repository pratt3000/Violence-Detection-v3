import config

import os
import cv2
import random
import numpy as np

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
