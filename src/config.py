import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_SIZE = 64
FRAME_BATCH_SIZE = 4
BATCH_SIZE = 16
EPOCHS = 5
TRAIN_DIR = "/home/pratt3000/Documents/projects/Violence Detection-v3/Data/"
VAL_DIR = "/home/pratt3000/Documents/projects/Violence Detection-v3/Data/"
MODEL_FILE_SAVE_PATH = "/home/pratt3000/Documents/projects/Violence Detection-v3/models/classification/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
TRAIN_DATASET_SIZE = 100
VAL_DATASET_SIZE= 10