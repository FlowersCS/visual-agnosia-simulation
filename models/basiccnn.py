import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torch.nn.utils.prune as prune

class BasicCNN(pl.LightningModule):
    
    def __init__(self, in_channels=3, num_classes=30, lr=3e-4, pruning=None):
        super(BasicCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        # Definir una arquitectura de CNN más profunda
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Bloque adicional para mayor complejidad
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * (224 // 32) * (224 // 32), 1024)  # Ajustado para img_size=224 y 5 capas de MaxPooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Función de pérdida y métricas
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def apply_pruning(self, amount, layers='initial'):
        print(f"Applying pruning with amount={amount} on {layers} layers.")
        modules = list(self.children())[:5] if layers == "initial" else list(self.children())[-5:]
        for name, module in enumerate(modules):
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                print(f"Pruned layer {name} with {amount*100}% of weights removed.")
        
        # Remover máscaras de poda para simplificar el modelo final
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.test_acc(torch.argmax(preds, dim=1), y)
        self.log('test_acc', self.test_acc, on_epoch=True)
