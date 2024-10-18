import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from models.resnet50 import resnet50
#from models.ViT import vit
from datamodule import DataModule
from utils.lightning_utils import configure_num_workers

def evaluate_model(checkpoint_path, data_dir, img_size=224, batch_size=32):
    # Cargar el modelo desde el checkpoint
    model = resnet50.load_from_checkpoint(checkpoint_path=checkpoint_path)
    #model = vit.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # Crear el DataModule con los datos de prueba
    datamodule = DataModule(
        name="Pictograms",
        img_size=img_size,
        img_channels=3,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=configure_num_workers()
    )
    
    # Preparar los datos de prueba
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    # Crear el entrenador solo para la evaluación
    #trainer = Trainer(accelerator="cpu", max_epochs=1)
    trainer = Trainer(max_epochs=1, accelerator="auto", devices="auto")

    # Evaluar el modelo en los datos de prueba
    trainer.test(model, datamodule.test_dataloader())
    
if __name__ == "__main__":
    # El path del checkpoint guardado durante el entrenamiento
    checkpoint_path = "experiments/resnet50/resnet50_0%_test4/last.ckpt"

    # Ruta al dataset (asegúrate de que es la ruta correcta al conjunto de prueba)
    data_dir = "dataset/pictograms"

    # Llamar a la función de evaluación
    evaluate_model(checkpoint_path, data_dir,img_size=224,batch_size=32)
