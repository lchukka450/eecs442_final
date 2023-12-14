import torch
import torch.nn as nn
import torch.optim as optim
import dataset 
import os 
from PIL import Image
import json
from tqdm import tqdm

from model import AcroModel
from dataset import PhysObsDataset
from torch.utils.data import DataLoader

def train_model(
        model: AcroModel,
        train_loader: DataLoader,
        # val_loader: DataLoader,
        epochs : int = 10,
        device : str='cuda'
):
    bce_criterion = nn.BCELoss()
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to_device(device)

    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch', leave=False) as pbar:
            for batch_idx, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                images.requires_grad = True

                optimizer.zero_grad()
                
                unary_labels = [torch.cat([i, j]).T.to(device) for i, j in zip(labels[0][0], labels[0][1])] # stack the labels for each image in pair
                unary_masks = [(unary_label != -1) for unary_label in unary_labels] # mask labels to 
                unary_labels = [unary_label[mask] for unary_label, mask in zip(unary_labels, unary_masks)]
                
                binary_labels = [label.to(device) for label in labels[1]]
                binary_masks = [(binary_label != 0) for binary_label in binary_labels]
                binary_labels = [binary_label[mask] for binary_label, mask in zip(binary_labels, binary_masks)]

                unary_outputs, binary_outputs = model(images)

                unary_loss = 0.0
                binary_loss = 0.0

                for i in range(len(unary_outputs)):
                    outputs = unary_outputs[i][unary_masks[i], :]
                    if outputs.nelement() != 0:
                        unary_loss += ce_criterion(outputs.float(), unary_labels[i].long())
                
                for i in range(len(binary_outputs)):
                    outputs = binary_outputs[i][binary_masks[i], :]
                    if outputs.nelement() != 0:
                        binary_loss += ce_criterion(outputs.float(), (binary_labels[i]==1).long())
                
                loss = unary_loss + binary_loss
                running_loss += loss.item()
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()
                pbar.set_description(f'Epoch {epoch} | Loss: {loss.item()}')
                pbar.update(1)
            print(f'Epoch {epoch} | Loss: {running_loss / len(train_loader)}')
            losses.append(running_loss / len(train_loader))

if __name__ == '__main__':
    dataset = PhysObsDataset('images')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = AcroModel(dinov2_vits14)
    train_model(model, dataloader)

    breakpoint()