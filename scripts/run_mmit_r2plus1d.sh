#!/bin/bash

DATA_PATH=path_to_data
PRETRAIN_PATH=path_to_pretrain_model
MMIT_VERSION=v1
M3A_MODE=GRAPH
M3A_MODAL_TYPE=AUDIO_TEXT
M3A_MODAL_JOINT_TYPE=SUM

python tools/run_net.py \
--cfg configs/Mmit/R2PLUS1D_${M3A_MODE}_8x2.yaml \
DATA.PATH_TO_DATA_DIR ${DATA_PATH} \
DATA.MMIT_VERSION ${MMIT_VERSION} \
TRAIN.CHECKPOINT_FILE_PATH ${PRETRAIN_PATH} \
M3A.MODE ${M3A_MODE} \
M3A.MODAL_TYPE ${M3A_MODAL_TYPE} \
M3A.MODAL_JOINT_TYPE ${M3A_MODAL_JOINT_TYPE}
