#!/bin/bash

MODEL_TYPE="Cnn14_DecisionLevelAtt"
CHECKPOINT_PATH="models/Cnn14_DecisionLevelAtt_mAP=0.425.pth"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "$CHECKPOINT_PATH exists."
else 
    echo "$CHECKPOINT_PATH does not exist. Downloading..."
    wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth?download=1"
fi

DATASET_PATH="/home/victor/Desktop/FUSA/IA/repositorios/training_datasets/datasets/UrbanSound8K/audio/all_audios/"
CSV_PATH="/home/victor/Desktop/FUSA/IA/repositorios/training_datasets/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
DATASET="UrbanSound"

python3 pytorch/inference_fusa.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audios_path=$DATASET_PATH \
    --meta_path=$CSV_PATH \
    --dataset_name=$DATASET \
    --cuda