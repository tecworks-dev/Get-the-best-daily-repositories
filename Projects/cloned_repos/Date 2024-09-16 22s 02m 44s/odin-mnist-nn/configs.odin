package main

// Files
MNIST_TEST_FILE_PATH :: "./data/mnist_test.csv"
MNIST_TRAIN_FILE_PATH :: "./data/mnist_train.csv"
NETWORK_SAVE_DIRECTORY :: "./res"
NETWORK_SAVE_FILE_PATH :: NETWORK_SAVE_DIRECTORY + "/net.json"
NETWORK_LOAD_FILE_PATH :: NETWORK_SAVE_DIRECTORY + "/net97.json"

// Mnist Data
MNIST_NUM_LABELS :: 10
MNIST_IMG_SIZE :: 28
MNIST_IMG_DATA_LEN :: MNIST_IMG_SIZE * MNIST_IMG_SIZE
TRAIN_DATA_LEN :: 60000
TEST_DATA_LEN :: 10000
DATA_AUGMENTATION_COUNT :: 5

// Train
BATCH_SIZE :: 50
NUM_STEPS :: 5000000
LEARNING_RATE :: 0.05
DROPOUT_RATE :: 0.01
NET_ARCH :: []u32 { 32, 24, 16 }

// Viz
WINDOW_W :: 1080
WINDOW_H :: 720
FPS :: 120
CONNECTION_LINES_THRESHOLD :: 150
WEIGHT_CLOUD_THRESHOLD :: 0.25
CAM_REVOLUTION_SPEED :: 0.6
CAM_REVOLUTION_RADIUS :: 60
BG_COLOR_DARK_BLUE :: 0x162432ff