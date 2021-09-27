#!/bin/bash

MODEL_TYPE="Wavegram_Logmel_Cnn14"

CHECKPOINT_PATH="models/Wavegram_Logmel_Cnn14_mAP=0.439.pth"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "$CHECKPOINT_PATH exists."
else 
    echo "$CHECKPOINT_PATH does not exist. Downloading..."
    wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1"
fi

DATASET_PATH="/home/victor/Desktop/FUSA/IA/repositorios/training_datasets/datasets/ESC-50/audio/"
CSV_PATH="/home/victor/Desktop/FUSA/IA/repositorios/training_datasets/datasets/ESC-50/meta/esc50.csv"
DATASET="ESC"

python3 pytorch/inference_fusa.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audios_path=$DATASET_PATH \
    --meta_path=$CSV_PATH \
    --dataset_name=$DATASET \
    --cuda