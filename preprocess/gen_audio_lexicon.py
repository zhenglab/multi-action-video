import os
import sys
import subprocess
import argparse
import random
import pickle
import collections
import numpy as np
import torch

def gen_audio_lexicon_v1(label_file, label_num, audio_feature_file, audio_base_output_file):
    print("gen_audio_lexicon_v1 triggered ......")

    audio_feature_dict = read_pkl(audio_feature_file)
    label_audio_dict = collections.OrderedDict()
    for i in range(label_num):
        label_audio_dict[i] = []
    
    miss_audio_count = 0
    with open(label_file, "r") as audio_labels:
        for line in audio_labels:
            line = line.strip("\n").strip()
            video_path, labels = line.split(',', 1)
            audio_path = os.path.splitext(video_path.strip())[0] + '.wav'
            label_list = [int(i.strip()) for i in labels.split(",")]
            if audio_path in audio_feature_dict:
                for label_i in label_list:
                    label_audio_dict[label_i].append(audio_path)
            else:
                miss_audio_count = miss_audio_count + 1

    for key in label_audio_dict.keys():
        print("{} - length: {}".format(str(key), len(label_audio_dict[key])))
    
    print("miss_audio_count - {}".format(miss_audio_count))
    print("audio_lexicon_count - {}".format(len(audio_feature_dict)))
    
    audio_base_dict = collections.OrderedDict()    
    audio_base_dict['label'] = label_audio_dict
    audio_base_dict['audio'] = audio_feature_dict
     
    with open(audio_base_output_file, 'wb') as f:
        pickle.dump(audio_base_dict, f, protocol=4)

def gen_audio_lexicon_v2(label_file, label_num, audio_feature_file, audio_base_output_file):
    print("gen_audio_lexicon_v2 triggered ......")

    audio_feature_dict_v1 = read_pkl(audio_feature_file)
    label_audio_dict_v2 = collections.OrderedDict()
    for i in range(label_num):
        label_audio_dict_v2[i] = []
    
    miss_audio_count = 0
    audio_feature_dict_v2 = collections.OrderedDict()
    with open(label_file, "r") as audio_labels:
        for line in audio_labels:
            line = line.strip("\n").strip()
            video_path, labels = line.split(',', 1)
            audio_path = os.path.splitext(video_path.strip())[0] + '.wav'
            label_list = [int(i.strip()) for i in labels.split(",")]
            if audio_path in audio_feature_dict_v1['audio']:
                for label_i in label_list:
                    label_audio_dict_v2[label_i].append(audio_path)
                audio_feature_dict_v2[audio_path] = audio_feature_dict_v1['audio'][audio_path]
            else:
                miss_audio_count = miss_audio_count + 1

    for key in label_audio_dict_v2.keys():
        print("{} - length: {}".format(str(key), len(label_audio_dict_v2[key])))
    
    print("miss_audio_count - {}".format(miss_audio_count))
    print("audio_lexicon_count - {}".format(len(audio_feature_dict_v2)))
    
    audio_base_dict = collections.OrderedDict()    
    audio_base_dict['label'] = label_audio_dict_v2
    audio_base_dict['audio'] = audio_feature_dict_v2
     
    with open(audio_base_output_file, 'wb') as f:
        pickle.dump(audio_base_dict, f, protocol=4)

if __name__ == '__main__':
#     1. gen audio base v1
    label_file = "trainingSet-v1-mini.txt"
    label_num = 313
    audio_feature_file = "mmit_v1_mini_audio.pkl"
    audio_base_output_file = "mmit_audio_lexicon_9w_v1.pkl"
    gen_audio_lexicon_v1(label_file, label_num, audio_feature_file, audio_base_output_file)
    
#     2. gen audio base v1
    label_file = "trainingSet-v2.txt"
    label_num = 292
    audio_feature_file = "mmit_audio_lexicon_9w_v1.pkl"
    audio_base_output_file = "mmit_audio_lexicon_9w_v2.pkl"
    gen_audio_lexicon_v2(label_file, label_num, audio_feature_file, audio_base_output_file)

