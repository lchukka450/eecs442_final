from dataset import PhysObsDataset
import json
from tqdm import tqdm
import PIL
import matplotlib.pyplot as plt
from dataset import DEFAULT_TRANSFORM
import torch
from model import AcroModel

dataset = PhysObsDataset("images")


dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = AcroModel(dinov2_vits14)
model.load_state_dict(torch.load("model.pth"))

with open('ego_objects_challenge_train.json', 'r') as file:
    json_data = json.load(file)

object_dict = {}

for fname in tqdm(dataset.filenames):
    train_image_id = dataset.fname_to_train_id[fname]
    image_entry = next((image for image in json_data["images"] if image["id"] == train_image_id), None)

    main_category = image_entry["main_category"]

    if main_category not in object_dict:
        object_dict[main_category] = [fname]
    else:
        object_dict[main_category].append(fname)

objects = ["Box of Macaroni & Cheese", "Christmas tree", "Digital clock", "Chest of drawers", "basketball"]
for object in tqdm(objects):
    
    object_images = [DEFAULT_TRANSFORM(PIL.Image.open(f"images/{image}")) for image in object_dict[object]]
    object_images = torch.stack(object_images)

    features = model.encode(object_images)
    torch.save(features, f"features/{object}.pth")
    