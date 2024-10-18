import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from torchvision import models

class vit(pl.LightningModule):
    
    def __init__(self, pretrained=True, in_channels=3, num_classes=16, lr=3e-4, freeze=False):
        super(vit, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        # Cargar el modelo Vision Transformer (ViT) de torchvision
        if pretrained:
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.model = models.vit_b_16(weights=None)

        in_features = self.model.heads[0].in_features

        # Modificar la capa de salida final (heads) para ajustarse al número de clases
        self.model.heads = nn.Linear(in_features, num_classes)
        
        if freeze:
            print("Congelando capas del modelo preentrenado para fine-tuning")
            for param in self.model.parameters():
                param.requires_grad = False

        # Función de pérdida
        self.loss_fn = nn.CrossEntropyLoss()

        # Métricas de entrenamiento, validación y test
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        
        self.train_acc(torch.argmax(preds, dim=1), y)
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        
        self.val_acc(torch.argmax(preds, dim=1), y)
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        
        self.test_acc(torch.argmax(preds, dim=1), y)
        self.log('test_acc', self.test_acc, on_epoch=True)
