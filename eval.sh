#!/usr/bin/env bash

GPUS=-1
model=ht
name=FCHT_2H9L_allihd

dataset_list=(ihd HAdobe5k HCOCO HFlickr Hday2night)
#dataset_root=/Users/zhenghui/Downloads/Image_Harmonization_Dataset
dataset_root=/workspace/dataset/Image_Harmonization_Dataset
dataset_mode=ihd
batch_size=16

for ds in "${dataset_list[@]}"
do
  echo Evaluate dataset: "${ds}"
  python test.py --light_use_mask --light_element 27 \
   --gpu_ids ${GPUS} \
   --name ${name} \
   --model ${model} \
   --dataset_mode ${dataset_mode} \
   --dataset_root ${dataset_root} \
   --dataset_name "${ds}" \
   --batch_size ${batch_size}
done
