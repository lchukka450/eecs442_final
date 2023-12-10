import torch
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset_extract
from PIL import Image

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((540, 1200)),  # Resize the image
    transforms.ToTensor(),          # Convert PIL image to tensor
    # Add more transformations if required
])

class PhysObsDataset(Dataset):
    def __init__(self, img_dir, phys_obs_json, transform=None):
        self.img_dir = img_dir
        self.filenames = glob.glob(f"{self.img_dir}/*.jpg")
        self.transform = transform if transform else DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        img = Image.open(fname)
        img = self.transform(img)
    
    

        