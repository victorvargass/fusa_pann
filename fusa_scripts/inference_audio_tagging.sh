#!/bin/bash

# ------ Inference audio tagging result with pretrained model. ------
MODEL_TYPE="Cnn14_DecisionLevelAtt"
CHECKPOINT_PATH="models/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
AUDIO_PATH="resources/dog_bark.wav"

python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path=$AUDIO_PATH \
    --cuda