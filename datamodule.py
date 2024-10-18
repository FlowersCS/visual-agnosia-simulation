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
from torch import tensor

class PictogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx,0]
        image = Image.open(image_path).convert('RGB')
        manner = self.dataframe.iloc[idx,1]
        obj = self.dataframe.iloc[idx,2]

        if self.transform:
            image = self.transform(image)
        
        return image, tensor(int(obj))#(manner, obj)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        img_size: int,
        img_channels: int,
        data_dir: Union[str, Path] = DATASET_PATH,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_split: float = 0.8,
        download: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.name = str(name)
        self.img_size = img_size
        self.img_channels = img_channels
        self.data_dir = data_dir
        self.batch_size = int(batch_size / (torch.cuda.device_count() if torch.cuda.device_count() > 1 else 1))
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.download = download
        self.pin_memory = pin_memory
        self.df = None

        self.transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5] * self.img_channels,
                        [0.5] * self.img_channels,
                    ),
                    #CenterCropMinXY(),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5] * self.img_channels,
                        [0.5] * self.img_channels,
                    ),
                    #CenterCropMinXY(),
                    
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((self.img_size,self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5] * self.img_channels,
                        [0.5] * self.img_channels,
                    ),
                    #CenterCropMinXY(),
                    
                ]
            ),
        }
    
    def prepare_data(self) -> None:
        if self.name == 'Pictograms':
            images = []
            manners = []
            objects = []

            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    if not file.lower().endswith('.png'):
                        print(f"Ignorando archivo no válido: {file_path}")
                        continue

                    manner = file[:2]
                    obj = file[3:6]

                    images.append(file_path)
                    manners.append(manner)
                    objects.append(obj)

            if images:
                self.df = pd.DataFrame({
                    'imagen': images,
                    'manner': manners,
                    'object': objects
                })

                print(f"Total imágenes en el dataset: {len(self.df)}")
            
                # Mapping labels to number value
                manner_map = {
                    '01': 'pictogram',
                    '02': 'contour',
                    '03': 'sketch'
                }

                #object_map = {
                #    '001': 'flower', '002': 'bird', '003': 'butterfly', '004': 'tree',
                #    '005': 'plane', '006': 'crane', '007': 'dog', '008': 'horse',
                #    '009': 'deer', '010': 'truck', '011': 'car', '012': 'cat',
                #    '013': 'frog', '014': 'ship', '015': 'fish', '016': 'house'
                #}
                object_map = {
                '001': 0, '002': 1, '003': 2, '004': 3, '005': 4, '006': 5, '007': 6, '008': 7,
                '009': 8, '010': 9, '011': 10, '012': 11, '013': 12, '014': 13, '015': 14, '016': 15
                }

                self.df['manner'] = self.df['manner'].replace(manner_map)
                self.df['object'] = self.df['object'].replace(object_map)
            
            else:
                print("No se encontraron imágenes válidas en el directorio.")
    
    def setup(self, stage: Optional[str] = None) -> None:
        if self.name == 'Pictograms':
            if self.df is None:
                raise ValueError("DataFrame no inicializado. Asegúrate de que prepare_data() se haya llamado.")
            # Crear el dataset
            full_dataset = PictogramDataset(self.df, transform=self.transforms['train'])

            # Dividir en entrenamiento y validación
            num_train = int(len(full_dataset) * self.train_val_split)
            num_val = len(full_dataset) - num_train
            self.train_dataset, self.val_dataset = random_split(full_dataset, [num_train, num_val])
            print(f"Tamaño del conjunto de entrenamiento: {len(self.train_dataset)}")
            print(f"Tamaño del conjunto de validación: {len(self.val_dataset)}")
            # Test dataset (puedes ajustar según tus necesidades)
            self.test_dataset = PictogramDataset(self.df, transform=self.transforms['test'])
            print(f"Tamaño del conjunto de prueba: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    