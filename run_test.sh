#!/bin/bash

# Configuración común
CONFIG_PATH="configs/resnet50.json"
CKPT_PATH="experiments/resnet50_0%/last.ckpt"
PROJECT="Exposicion_1_proyecto"

# Definir porcentajes y tipos de capas
pruning_amounts=(0.2 0.3 0.5 0.7)
layers=("initial" "final")

# Ejecutar los experimentos
for layer in "${layers[@]}"; do
  for amount in "${pruning_amounts[@]}"; do
    experiment_name="resnet50_${amount*100}%_${layer}"
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
