#!/bin/bash


### DeepCAD VAE Inference ###
python inference.py \
    --data data_process/deepcad_parsed \
    --train_list data_process/deepcad_data_split_6bit.pkl \
    --val_list data_process/deepcad_data_split_6bit.pkl \
    --test_list data_process/deepcad_data_split_6bit.pkl \
    --option surface \
    --gpu 0 \
    --env deepcad_vae_surf \
    --finetune \
    --weight /home/szj/BrepGen/checkpoints/deepcad_vae_surf.pt