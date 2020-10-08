import config
import functions as FUNC 
 
import random
import os
import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from keras.layers import Input, Permute, GaussianNoise, ConvLSTM2D, Conv2D, Flatten, Dense
from keras.models import Model 
from keras.callbacks import ModelCheckpoint

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(config.PRETRAINED_MODEL_PATH)
detector.loadModel(detection_speed = config.OBJECT_DETECTION_SPEED) #change parameter to adjust accuracy and speed
custom = detector.CustomObjects(person=True)

total_number_frames_train = FUNC.get_total_frames(config.TRAIN_DIR, config.TRAIN_DATASET_SIZE)
total_number_frames_valid = FUNC.get_total_frames(config.VAL_DIR, config.VAL_DATASET_SIZE)

steps_per_epoch = total_number_frames_train // (config.BATCH_SIZE * config.FRAME_BATCH_SIZE)
validation_steps = total_number_frames_valid // (config.BATCH_SIZE * config.FRAME_BATCH_SIZE)

def get_boxes(frame):
    _, detections = detector.detectCustomObjectsFromImage(
        custom_objects=custom,
        input_type="array",
        input_image= frame,
        output_type="array"
    )
    return detections


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
            
            detections_temp_2=[]
            images_frame_batches=[]
            
            success,frame_temp = vidcap.read()
            if success == False:
                break

            frame_temp = cv2.resize(frame_temp, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            #detections_temp = get_boxes(frame_temp) 
            #frame_temp = FUNC.mask_frame(frame_temp, detections_temp, detections_temp_2)                    
            frame_temp = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2GRAY)
            
            for k in range(config.FRAME_BATCH_SIZE - 1):

                success, frame = vidcap.read()
                if success == False:
                    flag_valid_video = 1
                    break
                    
                frame = cv2.resize(frame, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                #detections = get_boxes(frame)                   
                #frame = FUNC.mask_frame(frame, detections, detections_temp)   
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                  
                 
                diff = FUNC.get_frame_difference(frame, frame_temp)/255
                images_frame_batches.append(diff)

                frame_temp = frame
                #detections_temp=detections

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


inp = Input((config.FRAME_BATCH_SIZE - 1, config.IMG_SIZE, config.IMG_SIZE, 1))
permuted = Permute((2, 3, 4, 1))(inp)
noise = GaussianNoise(0.1)(permuted)
c = 4
x = Permute((4, 1, 2, 3))(noise)
conv_lstm_output_1 = ConvLSTM2D(6, (3, 3), padding='same')(x)
conv_output = Conv2D(3, (3, 3), padding="same")(conv_lstm_output_1)
# x =(ConvLSTM2D(filters=c, kernel_size=(3,3),padding='same',name='conv_lstm1', return_sequences=True))(x)

# c1=(BatchNormalization())(x)
# c1 = Dropout(0.9)(c1)
# x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c1)

# x =(ConvLSTM2D(filters=2c,kernel_size=(3,3),padding='same',name='conv_lstm2',return_sequences=True))(x)
# c2=(BatchNormalization())(x)
# x = Dropout(0.2)(x)

# x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c2)
# x =(ConvLSTM2D(filters=4c,kernel_size=(3,3),padding='same',name='conv_lstm3',return_sequences=True))(x)
# x =(BatchNormalization())(x)

# x = Add()([c2, x])
# x = Dropout(0.2)(x)

# x =(TimeDistributed(UpSampling2D(size=(2, 2))))(c1)
# x =(ConvLSTM2D(filters=c,kernel_size=(3,3),padding='same',name='conv_lstm6',return_sequences=False))(x)
# x =(BatchNormalization())(x)

x = (Flatten())(conv_output)
# x = (Dense(units=10, activation='relu'))(x)

x = (Dense(units=2, activation='relu'))(x)

model = Model(inputs=[inp], outputs=[x])
   

############################################### MODEL END ##########################################

print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

checkpoint = ModelCheckpoint(config.MODEL_FILE_SAVE_PATH, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

model.fit_generator(
                    my_generater(config.TRAIN_DIR, config.TRAIN_DATASET_SIZE),
                    steps_per_epoch = int(steps_per_epoch),
                    epochs = config.EPOCHS,
                    validation_steps = int(validation_steps),
                    validation_data = my_generater(config.VAL_DIR, config.VAL_DATASET_SIZE),
                    callbacks = [checkpoint]
                   )