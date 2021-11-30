#!/bin/bash

ANACONDA_HOME=$(pwd)/../../../anaconda38-tensorflow

export PATH=${ANACONDA_HOME}/bin:$PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda

CUDA_VISIBLE_DEVICES=0 python vggish/mmit_vggish_features.py \
--audio_dir /mnt/8TDisk1/Multi_Moments_in_Time_audios \
--label_file trainingSet-v1-mini.txt \
--write_file mmit_v1_mini_audio.pkl

CUDA_VISIBLE_DEVICES=0 python gen_audio_lexicon.py

