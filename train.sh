#!/usr/bin/env bash

GPUS=0,1,2,3
model=ht
name=ht_v1.0
niter=30
niter_decay=30
#dataset_root=/Users/zhenghui/Downloads/Image_Harmonization_Dataset
dataset_root=/workspace/dataset/Image_Harmonization_Dataset
dataset_name=ihd
batch_size=8

python train.py  --tr_r_enc_head 2 --tr_r_enc_layers 9 --light_element 27 \
 --model ${model} \
 --gpu_ids ${GPUS} \
 --name ${name} \
 --niter ${niter} \
 --niter_decay ${niter_decay} \
 --dataset_root ${dataset_root} \
 --dataset_name ${dataset_name} \
 --batch_size ${batch_size} \
 --init_port 9876 \
# --continue-train