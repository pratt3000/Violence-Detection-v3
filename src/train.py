import config
import functions as FUNC 
 
import random
import os
import cv2
import numpy as np
from keras.layers import Conv3D, MaxPooling3D, Activation, Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

total_number_frames_train = FUNC.get_total_frames(config.TRAIN_DIR, config.TRAIN_DATASET_SIZE)
total_number_frames_valid = FUNC.get_total_frames(config.VAL_DIR, config.VAL_DATASET_SIZE)

steps_per_epoch = total_number_frames_train // (config.BATCH_SIZE * config.FRAME_BATCH_SIZE)
validation_steps = total_number_frames_valid // (config.BATCH_SIZE * config.FRAME_BATCH_SIZE)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); 
fgbg = cv2.createBackgroundSubtractorMOG2();

def my_generater(in_dir, total_videos):     

    list_fight=os.listdir(os.path.join(in_dir, "Violence"))
    list_no_fight=os.listdir(os.path.join(in_dir, "NonViolence"))

    fight_final=random.sample(list_fight, total_videos//2)
    no_fight_final=random.sample(list_no_fight, total_videos//2)

    fight_labels = []
    no_fight_labels = []
    for i in range (total_videos//2):
        fight_labels.append([1, 0])
        no_fight_labels.append([0, 1])

    final = fight_final + no_fight_final
    labels = fight_labels + no_fight_labels

    c = list(zip(final, labels))
    random.shuffle(c)
    names, labels = zip(*c)
    
    images_batches=[]
    labelss=[]
    counter = 0
    
    while True: #select one video per loop

        video_number = random.randint(1,total_videos-1)

        if labels[video_number]==[1, 0]:
            in_file = os.path.join(in_dir, "Violence")
        else:
            in_file = os.path.join(in_dir, "NonViolence")

        in_file = os.path.join(in_file, names[video_number])
        vidcap = cv2.VideoCapture(in_file)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  #frames in a video

        flag_valid_video = 0

        for j in range(int(length/config.FRAME_BATCH_SIZE)):
            
            images_frame_batches=[]
            
            for k in range(config.FRAME_BATCH_SIZE - 1):

                success, frame = vidcap.read()
                if success == False:
                    flag_valid_video = 1
                    break
                    
                frame = cv2.resize(frame, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                frame = fgbg.apply(frame);
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel);                  
                frame = frame/255

                images_frame_batches.append(frame)

            if flag_valid_video == 1:
                labelss=[]
                images_batches=[]
                counter = 0
                break              

            if counter < config.BATCH_SIZE:
                images_batches.append(images_frame_batches)
                counter = counter + 1
                labelss.append(labels[video_number])

            else:
                yield np.array(images_batches).reshape((config.BATCH_SIZE, config.FRAME_BATCH_SIZE-1, config.IMG_SIZE, config.IMG_SIZE, 1)), np.array(labelss)
                labelss=[]
                images_batches=[]
                counter = 0




######################################## MODEL ###########################################

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(config.FRAME_BATCH_SIZE - 1, config.IMG_SIZE, config.IMG_SIZE, 1), border_mode='same'))
model.add(Activation('relu'))
model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
model.add(Dropout(0.25))
model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
model.add(Activation('relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())

############################################### MODEL END ##########################################

model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

checkpoint = ModelCheckpoint(config.MODEL_FILE_SAVE_PATH, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

model.fit_generator(
                    my_generater(config.TRAIN_DIR, config.TRAIN_DATASET_SIZE),
                    steps_per_epoch = int(steps_per_epoch),
                    epochs = config.EPOCHS,
                    validation_steps = int(validation_steps),
                    validation_data = my_generater(config.VAL_DIR, config.VAL_DATASET_SIZE),
                    callbacks = [checkpoint]
                   )

