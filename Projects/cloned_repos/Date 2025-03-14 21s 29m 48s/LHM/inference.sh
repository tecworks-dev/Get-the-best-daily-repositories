#!/bin/bash
# given pose sequence, generating animation video .

TRAIN_CONFIG="./configs/inference/human-lrm-1B.yaml"
MODEL_NAME="./exps/releases/video_human_benchmark/human-lrm-1B/step_060000/"
IMAGE_INPUT="./train_data/example_imgs/"
MOTION_SEQS_DIR="./train_data/motion_video/mimo6/smplx_params/"

TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"

MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=False

python -m LHM.launch infer.human_lrm --config $TRAIN_CONFIG \
        model_name=$MODEL_NAME image_input=$IMAGE_INPUT \
        export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS