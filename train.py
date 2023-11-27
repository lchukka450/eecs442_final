import torch
import torch.nn as nn
import torch.optim as optim
import dataset 
import os 
from PIL import Image

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ) 
    
    def forward(self, x):
        encoded_representation = self.encoder(x)
        return encoded_representation

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # decoder layers 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # Define forward pass for the decoder
        reconstructed_image = self.decoder(x)
        return reconstructed_image

encoder = Encoder()
decoder = Decoder()

criterion = nn.MSELoss()

# hyperparams 
learning_rate = 0.001
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
dataloader = dataset.get_dataloader()

num_epochs = 10
for epoch in range(num_epochs):
    
    for images, attributes in dataloader:  
        optimizer.zero_grad()
        
        encoded = encoder(images)
        
        reconstructed = decoder(encoded)
        loss = criterion(reconstructed, images)  
        
        # backwards pass and optimization
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

img_test = Image.open(os.path.join(os.getcwd(), "test_image.jpg"))
latent_representation = encoder(img_test)
