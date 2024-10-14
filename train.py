import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch


import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
from face_custom_dataset import FaceLandmarksDataset
from face_classification import FaceClassification
from utility import get_transformation
from utility import save_weights
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform,test_transform=get_transformation()
    train_face_dataset = FaceLandmarksDataset(
        root_dir="./dataset",
        transform=train_transform
    )
    test_face_dataset = FaceLandmarksDataset(
        root_dir="./dataset",
        transform=test_transform
    )
    num_cores = os.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    print(train_face_dataset.df.head())





    train_loader = DataLoader(train_face_dataset, batch_size=32, shuffle=True,num_workers=4,persistent_workers=True)
    val_loader = DataLoader(train_face_dataset, batch_size=32, shuffle=False,num_workers=4,persistent_workers=True)

    # Initialize the model
    model = FaceClassification(train_face_dataset.num_of_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        filename="face-recognition-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=6,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=False
    )

    # Initialize the logger
    logger = TensorBoardLogger(save_dir="lightning_logs", name="face_classification")

    # Initialize the Trainer
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )
    print(f"Number of workers in train_loader: {train_loader.num_workers}")
    print(f"Number of workers in val_loader: {val_loader.num_workers}")

    # Train the model
    ckpt_path= "./checkpoints/face-recognition-epoch=07-val_loss=0.01-val_acc=1.00.ckpt"
    trainer.fit(model, train_loader, val_loader)
    best_model_path = checkpoint_callback.best_model_path  # Path to the best checkpoint
    save_weights(best_model_path)



