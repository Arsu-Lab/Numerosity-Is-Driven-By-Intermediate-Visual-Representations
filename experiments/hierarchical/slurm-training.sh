#!/bin/bash

#SBATCH --job-name=1
#SBATCH --output=logs/1.out
#SBATCH --error=logs/1.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00

# 1. Conda env - New packages installed?
# 2. Wandb token added?
EXPERIMENT_PATH="/scratch/modelrep/sadiya/students/elias/hierarchical_experiment"
RENDER_PATH="/scratch/modelrep/sadiya/studets/elias/Renders"
OBJ_FUNCTION="varying_size"
CROSS_VAL_FUNCTION=""
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

  if [ "$MODEL" = "CORnetS" ] || [ "$MODEL" = "ResNet18" ]; then
      BLOCKS=4
  else
      BLOCKS=5
  fi

  for block in $(seq 1 $BLOCKS)
  do
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

      for i in {1..2}
      do
      	  export WANDB_MODEL_ID="{$MODEL_ID}-RUN-{$i}"
          echo "Block: $block, Iteration: $i, Model ID: $MODEL_ID"
          python3 "./experiments/hierarchical/train_with_backbone.py" $1 $2 $3 $4 $5 $MODEL_ID $6 $7 $block $i $8 $9
      done

  done
}

export -f train_model
export MIOPEN_DISABLE_CACHE=1 # For multi gpu training disable ROCm caches

# Call the function 4 times with different arguments
# MODEL IDS: CORnetS VGG19 ResNet18 AlexNet
mkdir -p ./logs

RUN=1

MODELID="CORnetS"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 0" > ./logs/${OBJ_FUNCTION}_T_T_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_T_T_${MODELID}_R_${RUN}.err & # Pretrained frozen
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'False' $MODELID 1" > ./logs/${OBJ_FUNCTION}_T_F_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_T_F_${MODELID}_R_${RUN}.err & # Pretrained unfrozen
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'False' 'True' $MODELID 2" > ./logs/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'False' 'False' $MODELID 3" > ./logs/${OBJ_FUNCTION}_F_F_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_F_F_${MODELID}_R_${RUN}.err & # Untrained unfrozen

MODELID="AlexNet"
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'True' $MODELID 4" > ./logs/${OBJ_FUNCTION}_T_T_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_T_T_${MODELID}_R_${RUN}.err & # Pretrained frozen
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'True' 'False' $MODELID 5" > ./logs/${OBJ_FUNCTION}_T_F_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_T_F_${MODELID}_R_${RUN}.err & # Pretrained unfrozen
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'False' 'True' $MODELID 6" > ./logs/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_F_T_${MODELID}_R_${RUN}.err & # Untrained frozen
srun --cpus-per-task=8 -n1 -N1 bash -c "train_model $EXPERIMENT_PATH $RENDER_PATH $OBJ_FUNCTION $CROSS_VAL_FUNCTION $BACKGROUND 'False' 'False' $MODELID 7" > ./logs/${OBJ_FUNCTION}_F_F_${MODELID}_R_${RUN}.log 2> ./logs/${OBJ_FUNCTION}_F_F_${MODELID}_R_${RUN}.err & # Untrained unfrozen
wait
