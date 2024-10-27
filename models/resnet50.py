import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from torchvision import models
import torch.nn.utils.prune as prune

class resnet50(pl.LightningModule):
    
    def __init__(self, pretrained=True, in_channels=3, num_classes=30, lr=3e-4, freeze=False, pruning=None):
        super(resnet50, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr
        
        
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = models.resnet50(weights=None)

        # Congelar capas si es necesario
        if freeze:
            print("Congelando capas del modelo preentrenado para fine-tuning")
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        
        ## Aplicar poda si está configurada
        #if pruning and pruning.get("enabled", False):
        #    self.apply_pruning(amount=pruning["amount"], layers=pruning["layers"])

        # Cambiar la capa fc de ResNet para ajustarse a 128 características intermedias
        #self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        
        # Añadir más capas después de la capa fc original
        #self.fc = nn.Sequential(
        #    nn.Dropout(0.3),
        #    nn.Linear(128, self.num_classes)
        #)
        
        
        # Función de pérdida
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Métricas de entrenamiento, validación y test
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def apply_pruning(self, amount, layers='initial'):
        print(f"Applying pruning with amount={amount} on {layers} layers.")
        if layers == "initial":
            # Podar los primeros 3 bloques residuales (capas iniciales)
            for name, module in list(self.model.named_modules())[:32]:  # Ajusta el rango de acuerdo a los bloques iniciales
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                    print(f"Pruned {name} with {amount*100}% of weights removed.")
        elif layers == 'final':
            # Podar las capas finales (últimos módulos)
            for name, module in list(self.model.named_modules())[-32:]:  # Ajusta el rango de acuerdo a los bloques finales
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                    print(f"Pruned {name} with {amount*100}% of weights removed.")
                    
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

    def forward(self, x):
        #x = self.model(x)
        #x = self.fc(x)
        return self.model(x)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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

"""
class resnet50Inference(pl.LightningModule):
    
    def __init__(self, pretrained=True):
        super(resnet50Inference, self).__init__()
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=1000)
    
    def apply_pruning(self, prune_percentage, layer_type="initial"):
        if prune_percentage > 0:
            layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]
            if layer_type == "initial":
                # Podar la primera capa (o varias capas iniciales según sea necesario)
                for i, layer in enumerate(layers[:1]):  # Cambia el 3 si quieres más capas
                    prune.l1_unstructured(layer, name='weight', amount=prune_percentage)
                    print(f"Podada capa inicial {i+1}, porcentaje {prune_percentage*100}%")
            elif layer_type == "final":
                # Podar la última capa (o varias capas finales)
                for i, layer in enumerate(layers[-1:]):  # Cambia el 3 si quieres más capas
                    prune.l1_unstructured(layer, name='weight', amount=prune_percentage)
                    print(f"Podada capa final {i+1}, porcentaje {prune_percentage*100}%")  

    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        
        # Calcular accuracy en el conjunto de test
        self.test_acc(torch.argmax(preds, dim=1), y)
        self.log('test_acc', self.test_acc, on_epoch=True)
    
    def configure_optimizers(self):
        # No optimizador ya que no estamos entrenando
        pass

    def on_test_start(self):
        # Configurar el modelo en modo evaluación antes del testeo
        self.model.eval()
        # Desactivar gradientes para la inferencia
        torch.set_grad_enabled(False)

"""