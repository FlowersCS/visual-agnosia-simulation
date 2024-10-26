import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from models.resnet50 import resnet50Inference
#from models.ViT import vit
from datamodule import DataModule
from utils.lightning_utils import configure_num_workers
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    
    # Argumentos principales
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the ImageNet validation dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=configure_num_workers(), help="Number of workers for DataLoader")
    #parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--weight_diminution", type=float, default=0, help="Percentage of weight pruning (0 for no pruning)")
    parser.add_argument("--prune_type", type=str, choices=["initial", "final"], default="initial", help="Prune the initial or final layers")

    return parser.parse_args()


def main():
    args = parse_args()

    model = resnet50Inference(pretrained=True)

    datamodule = DataModule(
        img_size=224,
        img_channels=3,
        data_dir=args.data_dir,
        batch_size= args.batch_size,
        num_workers=args.num_workers
    )

    datamodule.prepare_data()
    datamodule.setup(stage="test")

    trainer = pl.Trainer(
        max_epochs=1, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )

    trainer.test(model, dataloaders=datamodule.test_dataloader())

if __name__ == "__main__":
    main()