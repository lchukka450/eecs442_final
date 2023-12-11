import torch
import torch.nn as nn
import torch.optim as optim
import dataset 
import os 
from PIL import Image
import json

from model import AcroModel
from dataset import PhysObsDataset


def train_model(
        model,
        train_loader,
        val_loader,
        epochs,
        device
):
    bce_criterion = nn.BCELoss()
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            img1, img2, label = data
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = bce_criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100}')
                running_loss = 0.0