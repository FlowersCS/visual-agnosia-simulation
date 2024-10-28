import torch
import pytorch_lightning as pl
import os
import pandas as pd

from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
from utils.path import DATASET_PATH
from pathlib import Path
from torchvision import datasets, transforms
from typing import Optional, Union
from torch import Tensor
from torchvision.io import read_image

class CenterCropMinXY:
    """
    Custom transform that performs a center crop on the image in the smaller dimension (X or Y).
    """

    def __call__(self, image: Union[Tensor, any]) -> Tensor:
        """
        Perform the center crop on the image.

        :param image: The input image as a torch.Tensor.
        :return: The cropped image as a torch.Tensor.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError('Input image should be a torch.Tensor')

        # Get the height and width of the image
        _, h, w = image.shape

        # Determine the smaller dimension
        min_dim = min(h, w)

        # Calculate top and left coordinates for cropping
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2

        # Perform the crop
        image = image[:, top: top + min_dim, left: left + min_dim]

        return image

class plantsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=0)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        data_dir: Union[str, Path] = DATASET_PATH,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_val_split: float= 0.8,
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.data_dir = Path(data_dir)
        self.batch_size = int(batch_size / (torch.cuda.device_count() if torch.cuda.device_count() > 1 else 1))
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_val_split = train_val_split

        self.transforms = {
            "train": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),       # Volteo horizontal aleatorio con probabilidad 50%
                    transforms.RandomRotation(degrees=20),        # Rotación aleatoria hasta 20 grados
                    transforms.ColorJitter(brightness=0.2,        # Cambio aleatorio de brillo, saturación y contraste
                                        contrast=0.2,
                                        saturation=0.2),
                    transforms.RandomResizedCrop(size=(self.img_size, self.img_size), 
                                                scale=(0.8, 1.0)), # Recorte aleatorio con 80%-100% del área
                    transforms.ToTensor(),
                    CenterCropMinXY(),
                    transforms.Normalize(
                        [0.5] * self.img_channels,
                        [0.5] * self.img_channels,
                    ),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.ToTensor(),
                    CenterCropMinXY(),
                    transforms.Normalize(
                        [0.5] * self.img_channels,
                        [0.5] * self.img_channels,
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.ToTensor(),
                    CenterCropMinXY(),
                    transforms.Normalize(
                        [0.5] * self.img_channels,
                        [0.5] * self.img_channels,
                    ),
                ]
            ),
        }

    
    #def prepare_data(self) -> None:

    def setup(self, stage: Optional[str] = None) -> None:
        train_file = self.data_dir / "train.csv"
        val_file = self.data_dir / "val.csv"
        test_file = self.data_dir / "test.csv"

        # Inicializar datasets
        self.train_dataset = plantsDataset(
            annotations_file=train_file,
            img_dir=self.data_dir,
            transform=self.transforms["train"],
        )
        self.val_dataset = plantsDataset(
            annotations_file=val_file,
            img_dir=self.data_dir,
            transform=self.transforms["val"],
        )
        self.test_dataset = plantsDataset(
            annotations_file=test_file,
            img_dir=self.data_dir,
            transform=self.transforms["test"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )