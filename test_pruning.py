# test_pruning.py
import pytorch_lightning as pl
import argparse
import os
import yaml

from datetime import datetime
from pprint import pprint
from pathlib import Path

from pytorch_lightning import Trainer
from models.resnet50 import resnet50
from models.vit import vit
from datamodule import DataModule
from utils.seed import seed_everything
from utils.lightning_utils import configure_num_workers, configure_strategy
from utils.loader import load_config, load_model
from utils.path import EXPERIMENT_DIR
from pytorch_lightning.loggers import WandbLogger
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch

seed_everything(seed=10, workers=True)
EXPERIMENT_TIME = datetime.now().strftime("%Y-%m-%d_%H:%M")

def setup_arguments(print_args: bool = True, save_args: bool = True):
    parser = argparse.ArgumentParser("Test script with pruning")
    
    parser.add_argument("--config_path", type=str, required=True, help="Path to configs")
    parser.add_argument("--num_workers", type=int, default=configure_num_workers())
    parser.add_argument("--pruning_amount", type=float, required=True, help="Amount of weights to prune (e.g., 0.2 for 20%)")
    parser.add_argument("--layers_to_prune", type=str, choices=["initial", "final"], required=True, help="Layers to prune")
    parser.add_argument("--strategy", type=str, default=configure_strategy())
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument(
        "--project",
        type=str,
        default="Exposicion_1_proyecto",
        help="W&B project name.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=EXPERIMENT_TIME,
        help="W&B experiment name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume W&B.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="W&B run ID to resume from.",
    )

    args = parser.parse_args()
    args.config = load_config(args.config_path)

    args.experiment_dir = os.path.join(
        EXPERIMENT_DIR,
        args.config["model"]["name"],
        args.experiment_name
    )
    os.makedirs(args.experiment_dir, exist_ok=True)

    if print_args:
        pprint(vars(args))
    
    if save_args:
        config_name = Path(args.config_path).name
        config_path = os.path.join(args.experiment_dir, config_name)
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)
    
    return args

def apply_pruning(model, amount, layers):
    print(f"Applying pruning with amount={amount} on {layers} layers.")
    if amount > 0:  # Only apply pruning if amount is greater than 0
        if layers == "initial":
            for name, module in list(model.model.named_modules())[:32]:  # Ajusta a las capas iniciales
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                    print(f"Pruned {name} with {amount * 100}% of weights removed.")
        elif layers == "final":
            for name, module in list(model.model.named_modules())[-32:]:  # Ajusta a las capas finales
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                    print(f"Pruned {name} with {amount * 100}% of weights removed.")
        for name, module in model.model.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

if __name__ == "__main__":
    args = setup_arguments(print_args=True, save_args=True)

    # Cargar el modelo desde el checkpoint especificado
    model = load_model(args.config["model"])
    if args.ckpt_path:
        print(f"Loading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint["state_dict"])

    # No aplicar poda si `pruning_amount` es 0
    if args.pruning_amount > 0:
        apply_pruning(model, amount=args.pruning_amount, layers=args.layers_to_prune)

    # Configurar el DataModule solo para el conjunto de test
    datamodule = DataModule(
        **args.config["dataset"],
        num_workers=args.num_workers,
        pin_memory=True,
    )

    wandb_logger = WandbLogger(
        name=args.experiment_name,
        save_dir=args.experiment_dir,
        config=args.config["model"].update(args.config["dataset"]),
        project=args.project,
        resume="must" if args.resume else None,
        id=args.id if args.resume else None,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        strategy=args.strategy,
        precision=args.precision,
        deterministic=True,
    )

    # Realizar inferencia en el conjunto de test con la poda aplicada
    #print(f"Number of samples in test dataset: {len(datamodule.test_dataloader().dataset)}")
    trainer.test(model, datamodule=datamodule)
