import torch
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset_extract
from PIL import Image
import pandas as pd
from typing import NamedTuple
import numpy as np 
import json
import random

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((560, 560)),  # Resize the image
    transforms.ToTensor(),          # Convert PIL image to tensor
    # Add more transforpmations if required
])

class AcroObj:
    def __init__(self, id, acro_id, contains_liquid=None, is_sealed=None, material=None, transparency=None):
        self.id = id
        self.acro_id = acro_id
        self.contains_liquid = contains_liquid
        self.is_sealed = is_sealed
        self.material = material
        self.transparency = transparency
    def update(self, new_values):
        for k, v in new_values.items():
            setattr(self, k, v)
    def __str__(self):
        return ("AcroObj(id={}, acro_id={}, contains_liquid={}, is_sealed={}, "
                "material={}, transparency={})").format(self.id, self.acro_id, 
                                                         self.contains_liquid, 
                                                         self.is_sealed, 
                                                         self.material, 
                                                         self.transparency)
    

class PhysObsDataset(Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.filenames = glob.glob(f"{self.img_dir}/*.jpg")
        self.transform = transform if transform else DEFAULT_TRANSFORM

        self.acro_obj_list = {}
        self.deformability_matrix = None
        self.mass_matrix = None
        self.fragility_matrix = None 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index, as_tensor=False):

        fname = self.filenames[index]
        img = Image.open(fname)
        img = self.transform(img)

        return img
    
    def get_deformability_matrix(self):
        data = pd.read_csv("physobjects/annotations/crowdsourced/deformability/train.csv")
        id_dict = {}
        for i, row in data.iterrows():
            image_id_1 = int(row["annotation_id_0"])
            image_id_2 = int(row["annotation_id_1"])
            acro_id_1 = -1
            acro_id_2 = -1
            if image_id_1 not in self.acro_obj_list:
                self.acro_obj_list[image_id_1] = AcroObj(image_id_1, len(self.acro_obj_list), contains_liquid=row["response"])
            acro_id_1 = self.acro_obj_list[image_id_1].acro_id
            
            if image_id_2 not in self.acro_obj_list:
                self.acro_obj_list[image_id_2] = AcroObj(image_id_2, len(self.acro_obj_list), contains_liquid=row["response"])
            acro_id_2 = self.acro_obj_list[image_id_2].acro_id
            id_dict[(acro_id_1, acro_id_2)] = (row["response"]) 
        max_id = max(id_dict, key=lambda x: max(x))
        max_id = max(max_id[0], max_id[1])
        matrix = np.zeros((max_id + 1, max_id + 1))
        
        for key in id_dict.keys():
            # print(key)
            vals = id_dict[key]
            if (vals == "left"):
                matrix[key[0]] [key[1]] = 1
                matrix[key[1]] [key[0]] = -1
            else:
                matrix[key[0]] [key[1]] = -1
                matrix[key[1]] [key[0]] = 1

        self.deformability_matrix = matrix
        return matrix

    def get_fragility_matrix(self):
        data = pd.read_csv("physobjects/annotations/crowdsourced/fragility/train.csv")
        id_dict = {}
        for i, row in data.iterrows():
            image_id_1 = int(row["annotation_id_0"])
            image_id_2 = int(row["annotation_id_1"])
            acro_id_1 = -1
            acro_id_2 = -1
            if image_id_1 not in self.acro_obj_list:
                self.acro_obj_list[image_id_1] = AcroObj(image_id_1, len(self.acro_obj_list), contains_liquid=row["response"])
            acro_id_1 = self.acro_obj_list[image_id_1].acro_id
            
            if image_id_2 not in self.acro_obj_list:
                self.acro_obj_list[image_id_2] = AcroObj(image_id_2, len(self.acro_obj_list), contains_liquid=row["response"])
            acro_id_2 = self.acro_obj_list[image_id_2].acro_id
            id_dict[(acro_id_1, acro_id_2)] = (row["response"]) 
        max_id = max(id_dict, key=lambda x: max(x))
        max_id = max(max_id[0], max_id[1])
        matrix = np.zeros((max_id + 1, max_id + 1))
        
        for key in id_dict.keys():
            # print(key)
            vals = id_dict[key]
            if (vals == "left"):
                matrix[key[0]] [key[1]] = 1
                matrix[key[1]] [key[0]] = -1
            else:
                matrix[key[0]] [key[1]] = -1
                matrix[key[1]] [key[0]] = 1

        self.fragility_matrix = matrix
        return matrix

    def get_mass_matrix(self):
        data = pd.read_csv("physobjects/annotations/crowdsourced/mass/train.csv")
        id_dict = {}
        for i, row in data.iterrows():
            image_id_1 = int(row["annotation_id_0"])
            image_id_2 = int(row["annotation_id_1"])
            acro_id_1 = -1
            acro_id_2 = -1
            if image_id_1 not in self.acro_obj_list:
                self.acro_obj_list[image_id_1] = AcroObj(image_id_1, len(self.acro_obj_list), contains_liquid=row["response"])
            acro_id_1 = self.acro_obj_list[image_id_1].acro_id
            
            if image_id_2 not in self.acro_obj_list:
                self.acro_obj_list[image_id_2] = AcroObj(image_id_2, len(self.acro_obj_list), contains_liquid=row["response"])
            acro_id_2 = self.acro_obj_list[image_id_2].acro_id
            id_dict[(acro_id_1, acro_id_2)] = (row["response"]) 
        max_id = max(id_dict, key=lambda x: max(x))
        max_id = max(max_id[0], max_id[1])
        matrix = np.zeros((max_id + 1, max_id + 1))
        
        for key in id_dict.keys():
            # print(key)
            vals = id_dict[key]
            if (vals == "left"):
                matrix[key[0]] [key[1]] = 1
                matrix[key[1]] [key[0]] = -1
            else:
                matrix[key[0]] [key[1]] = -1
                matrix[key[1]] [key[0]] = 1

        self.mass_matrix = matrix
        return matrix

    def read_contains_liquid(self):
        data = pd.read_csv("physobjects/annotations/crowdsourced/can_contain_liquid/train.csv")
        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                self.acro_obj_list[image_id].update({'contains_liquid': row["response"]})
                # self.acro_obj_list[image_id] = self.acro_obj_list[image_id]._replace(contains_liquid=row["response"])
            else:
                self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), contains_liquid=row["response"])
                # self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), contains_liquid=row["response"])
    
    def read_is_sealed(self):
        data = pd.read_csv("physobjects/annotations/crowdsourced/is_sealed/train.csv")
        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                self.acro_obj_list[image_id].update({'is_sealed': row["response"]})
                # self.acro_obj_list[image_id] = self.acro_obj_list[image_id]._replace(is_sealed=row["response"])
            else:
                self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), is_sealed=row["response"])
                # self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), is_sealed=row["response"])
    
    def material(self):
        materials = {}
        data = pd.read_csv("physobjects/annotations/crowdsourced/material/train.csv")
        for i, row in data.iterrows():
            if (row["response"] not in materials):
                materials[row["response"]] = 1
            else:
                materials[row["response"]] += 1

        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                if (materials[row["response"]] > 71 and row["response"] != "unknown"):
                    self.acro_obj_list[image_id].update({'material': row["response"]})
                    # self.acro_obj_list[image_id] = self.acro_obj_list[image_id]._replace(material=row["response"])
            else:
                if (materials[row["response"]] > 71 and row["response"] != "unknown"):
                    self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), material=row["response"])
                # self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), material=row["response"])
        #print("NUM MATERIALS: ", materials)
    def transparency(self):
        data = pd.read_csv("physobjects/annotations/crowdsourced/transparency/train.csv")
        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                self.acro_obj_list[image_id].update({'transparency': row["response"]})
                # self.acro_obj_list[image_id] = self.acro_obj_list[image_id]._replace(transparency=row["response"])
            else:
                self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), transparency=row["response"])
                # self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), transparency=row["response"])
                
    def get_data(self):
        self.read_contains_liquid()
        self.read_is_sealed()
        self.material()
        self.transparency()
        return self.acro_obj_list
    
    def get_stats(self):
        matrix = np.zeros((4, 4))
        for obj in self.acro_obj_list.values():
            attrs = [obj.contains_liquid, obj.is_sealed, obj.material, obj.transparency]
            
            for i in range(4):
                for j in range(i+1, 4): 
                    if attrs[i] is not None and attrs[j] is not None:
                        matrix[i][j] += 1
                        matrix[j][i] += 1
        return matrix
        
        
# dataset = PhysObsDataset("", "", "")
# 
# Load JSON 
with open('ego_objects_challenge_train.json', 'r') as file:
    json_data = json.load(file)
train_image_ids = [2432626686873416, 487812089498173]
#train_image_ids = [image["id"] for image in json_data["images"][:20]]
#train_image_ids = random.sample([image["id"] for image in json_data["images"]], 20)
'''
with open('physobjects/instance_ids/train_ids.json', 'r') as file:
    json_ids = json.load(file)
train_image_ids = random.sample(json_ids, 20)
id_lookup = {entry["instance_id"]: entry["image_id"] for entry in json_data["annotations"]}
'''
'''
for inst_id in train_image_ids:
    id_value = id_lookup.get(inst_id)
    if id_value is not None:
        print(f"For Instance ID {inst_id}, the associated ID is: {id_value}")
    else:
        print(f"No matching ID found for Instance ID {inst_id}")
'''
       
for train_image_id in train_image_ids:
    image_entry = next((image for image in json_data["images"] if image["id"] == train_image_id), None)
    #id_value = id_lookup.get(train_image_id)
    #image_entry = next((image for image in json_data["images"] if image["id"] == id_value), None)
    if image_entry:
        main_category = image_entry["main_category"]
        print(f"The main_category for image {train_image_id} is: {main_category}")
        
        main_category_instance_ids = image_entry["main_category_instance_ids"]
        print(f"The main_category_instance_ids for image {train_image_id} is: {main_category_instance_ids[-1]}")
        
        group_id = image_entry["group_id"]
        print(f"The group_id for image {train_image_id} is: {group_id}")
          
        data = dataset.get_data()
        item = main_category_instance_ids[-1]
        if item in data:
            print(data[item])
        else:
            print("Item not found in categories")
    else:
        print(f"Image with ID {train_image_id} not found.")
     
#print(dataset.get_data()[243510])
## print(dataset.get_stats())

#print(dataset.get_mass_matrix())
