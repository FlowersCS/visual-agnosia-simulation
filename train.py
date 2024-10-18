# main.py
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.callbacks import LogArtifactCallback


seed_everything(seed=10, workers=True)
EXPERIMENT_TIME = datetime.now().strftime("%Y-%m-%d_%H:%M")

def setup_arguments(print_args: bool = True, save_args: bool = True):
    parser = argparse.ArgumentParser("Train script")
    
    parser.add_argument("--config_path", type=str, required=True, help="Path to configs")
    parser.add_argument("--num_workers", type=int, default=configure_num_workers())
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--strategy", type=str, default=configure_strategy())
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
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

if __name__ == "__main__":
    args = setup_arguments(print_args=True, save_args=True)

    model = load_model(args.config["model"])
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

    callbacks = [
        ModelCheckpoint(
            dirpath=args.experiment_dir,
            save_last=True,
            monitor= "val_acc",
        ),
        LogArtifactCallback(
            file_path=os.path.join(args.experiment_dir, Path(args.config_path).name),
        )
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        default_root_dir=args.experiment_dir,
        strategy=args.strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=args.precision,
        deterministic=True,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.ckpt_path
    )

"""
# MEJORAS RESNET50
- Frezze: False
- Capa final, resnet50. Adicionar capas fc
- Data augmentation en datamodule
- Regularizacion: drouput -> 0.4,0.5 si el modelo se ajuste demasiado rápido en el training

# MEJORAS VIT
- Freeze: False
- Batch mayor: batch_size -> 64, transformers tienden a funcionar mejor con batches grandes
- Optimizacion: AdamW -> Adam, LAMB
- img_size: 224x224 es el standar pero si se puede 384x384
- capa de normalizacion y preprocesamiento: Verifica que las imágenes se normalicen correctamente para adaptarse a la distribución del preentrenamiento de ViT (IMAGENET1K_V1).
    Asegúrate de que las transformaciones y normalización estén alineadas con el modelo preentrenado que estás usando.
- dificiles lr, funciona mejor cosine Annealing.

# GENERALES
- 3e-4 -> 1e-4
- Data augmentation
- batch size

# steps
- freeze
- lr y batch size
- capas fc en restnet50
- data augmentation in datamodule
- scheduler de tipo cosine annealing en ViT
"""