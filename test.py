from dataset import PhysObsDataset, DEFAULT_TRANSFORM
from model import AcroModel
import torch

dataset = PhysObsDataset('images')

dinov2_vitg14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
dinov2_vitg14_lc = dinov2_vitg14_lc.to('cuda')

model = AcroModel(dinov2_vitg14_lc).to('cuda')

img1 = dataset[0]
img2 = dataset[1]
imgs = torch.stack((img1, img2), dim=0).unsqueeze(0).to('cuda')

output = model(imgs)
for k, v in output.items():
    print(k, v.shape)