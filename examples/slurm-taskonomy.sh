#!/bin/bash

#SBATCH --job-name=10-taskonomy
#SBATCH --output=logs/taskonomy/1.out
#SBATCH --error=logs/taskonomy/1.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00

EXPERIMENT_PATH="/path/to/taskonomy"
RENDER_PATH="/path/to/renders"
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
      python3 "./examples/train_with_backbone.py" $1 $2 $3 $4 $5 $MODEL_ID $6 $7 $block $i $8 $9
  done
}

export -f train_model
export MIOPEN_DISABLE_CACHE=1 # For multi gpu training disable ROCm caches, otherwise possible race condition

# Call the function 4 times with different arguments
mkdir -p ./logs

RUN=10

MODELID="ResNet50Taskonomy-segment_unsup2d"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0" > ./logs/taskonomy/${OBJ_FUNCTION}_T_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_T_T_${MODELID}_R_${RUN}.err & # Pretrained frozen

MODELID="ResNet50Taskonomy-keypoints3d"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 1" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen

MODELID="ResNet50Taskonomy-class_object"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 2" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen

MODELID="ResNet50Taskonomy-depth_zbuffer"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 3" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen

MODELID="ResNet50Taskonomy-segment_semantic"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 4" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen

MODELID="ResNet50Taskonomy-reshading"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 5" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen

MODELID="ResNet50Taskonomy-edge_texture"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 6" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen

MODELID="ResNet50Taskonomy-segment_unsup25d"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 7" > ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/taskonomy/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen
wait