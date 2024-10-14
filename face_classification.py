import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import lightning as L
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class FaceClassification(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = InceptionResnetV1(num_classes=num_classes, classify=True)

    def forward(self, x):
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        map = batch
        x = map["image"]
        y = map["id"]
        y_hat = self(x)

        # Log images to TensorBoard
        grid = torchvision.utils.make_grid(x[:4])  # Select first 4 images from the batch
        self.logger.experiment.add_image(f"train_images", grid, self.global_step)

        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        map = batch
        x = map["image"]
        y = map["id"]
        y_hat = self(x)

        # Log images to TensorBoard
        grid = torchvision.utils.make_grid(x[:4])  # Select first 4 images from the batch
        self.logger.experiment.add_image(f"val_images", grid, self.global_step)

        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        map = batch
        x = map["image"]
        y = map["id"]
        y_hat = self(x)

        # Log images to TensorBoard
        grid = torchvision.utils.make_grid(x[:4])  # Select first 4 images from the batch
        self.logger.experiment.add_image(f"test_images", grid, self.global_step)

        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
