#!/bin/bash

# Adjust these paths to your own
EXPERIMENT_PATH="/home/elwa/research/taskonomy"
RENDER_PATH="/home/elwa/research/Renders"

OBJ_FUNCTION="varying_size_fixed_image_fixed_between"
CROSS_VAL_FUNCTION="varying_size_fixed_image_fixed_between_4val"
BACKGROUND="real"

train_model() {
  EXPERIMENT_PATH=$1
  RENDER_PATH=$2
  OBJC_FUNCTION=$3
  CROSS_VAL_FUNCTION=$4
  BACKGROUND=$5
  PRETRAINED=$6
  FREEZE_BACKGROUND=$7
  MODEL=$8
  ACC_NUMBER=$9

  block=4
  if [ "$PRETRAINED" = "True" ]; then
      PRETRAINED_LABEL="pretrained"
  else
      PRETRAINED_LABEL="untrained"
  fi

  if [ "$FREEZE_BACKGROUND" = "True" ]; then
      FREEZE_BACKGROUND_LABEL="frozen"
  else
      FREEZE_BACKGROUND_LABEL="unfrozen"
  fi

  MODEL_ID="$MODEL-$block-$PRETRAINED_LABEL-$FREEZE_BACKGROUND_LABEL"

  for i in {1..16}
  do
      export WANDB_MODEL_ID="{$MODEL_ID}-RUN-{$i}"
      echo "Block: $block, Iteration: $i, Model ID: $MODEL_ID"
      python3 "./experiments/hierarchical/train_with_backbone.py" $1 $2 $3 $4 $5 $MODEL_ID $6 $7 $block $i $8 $9
  done
}

export -f train_model


MODELID="ResNet50Taskonomy-autoencoding"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-class_object"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-class_scene"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-curvature"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-denoising"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-depth_euclidean"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-depth_zbuffer"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-edge_occlusion"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-edge_texture"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-inpainting"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-keypoints2d"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-keypoints3d"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-normal"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-reshading"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-segment_semantic"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-segment_unsup25d"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-segment_unsup2d"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"

MODELID="ResNet50Taskonomy-colorization"
train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0
echo "Finished training $MODELID"