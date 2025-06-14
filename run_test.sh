#!/bin/bash

# configuration
CONFIG_PATH="configs/resnet50.json" # | "configs/vit.json"
CKPT_PATH="experiments/resnet50/resnet50_0%/last.ckpt" # | "experiments/vit/vit_test3/last.ckpt"
PROJECT="Exposicion_1_proyecto"

# ckpt_path exist?
if [ ! -f "$CKPT_PATH" ]; then
  echo "Error: No se encontró el checkpoint en $CKPT_PATH"
  exit 1
fi

# layers && %weight diminution
pruning_amounts=(0.2 0.3 0.5 0.7)
layers=("initial" "final")

for layer in "${layers[@]}"; do
  for amount in "${pruning_amounts[@]}"; do
    percent=$(printf "%.0f" $(echo "$amount * 100" | bc -l 2>/dev/null || echo "$amount * 100"))

    experiment_name="cnn_${percent}%_${layer}"
    echo "Running experiment: $experiment_name with pruning_amount $amount on $layer layers"

    python3 test_pruning.py \
      --config_path $CONFIG_PATH \
      --pruning_amount $amount \
      --layers_to_prune $layer \
      --experiment_name $experiment_name \
      --ckpt_path $CKPT_PATH \
      --project $PROJECT
  done
done
