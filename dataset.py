import torch
from torch.utils.data import Dataset
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from typing import NamedTuple
import numpy as np 
import json
import os

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((560, 560)),   
    transforms.ToTensor(),           
])

class AcroObj:
    def __init__(self, id, acro_id, contains_liquid=-1, is_sealed=-1, material=-1, transparency=-1):
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
                
    def get_id(self):
        return self.id
    

class PhysObsDataset(Dataset):
    def __init__(self, images_dir, physobs_base_dir="physobjects/annotations/crowdsourced", transform=None, train=True):
        
        self.materials_enum = {}
        self.images_dir = images_dir
        self.phys_obs_base_dir = physobs_base_dir
        self.phys_obs_csv = "train" if train else "test"
        self.transform = transform if transform else DEFAULT_TRANSFORM

        self.acro_obj_list = {}
        self.deformability_matrix = self.get_deformability_matrix()
        self.mass_matrix = self.get_mass_matrix()
        self.fragility_matrix = self.get_fragility_matrix()

        max_acro_id = max(self.deformability_matrix.shape[0], self.mass_matrix.shape[0], self.fragility_matrix.shape[0])
        self.deformability_matrix = self.pad_matrix(self.deformability_matrix, max_acro_id)
        self.mass_matrix = self.pad_matrix(self.mass_matrix, max_acro_id)
        self.fragility_matrix = self.pad_matrix(self.fragility_matrix, max_acro_id)
            
        self.filenames, self.fname2acro, self.id2fname = self.read_phys_obs_data()
        self.acro_id2id = self.get_acro_to_id()

        self.candidate_pairs = list(self.get_pairs())
        self.pairs = []

        for pair in self.candidate_pairs:
            if pair[0] in self.acro_id2id and pair[1] in self.acro_id2id and self.acro_id2id[pair[0]] in self.id2fname and self.acro_id2id[pair[1]] in self.id2fname:
                self.pairs.append(pair)
        self.pairs = np.array(self.pairs)
        

    def __len__(self):
        return self.pairs.shape[0]
    
    def __getitem__(self, index):

        pair = self.pairs[index]
        
        ids = [self.acro_id2id[i] for i in pair]
        fnames = [self.id2fname[i] for i in ids]
        acros = [self.fname2acro[fname] for fname in fnames]

        imgs = [Image.open(os.path.join(self.images_dir, fname)) for fname in fnames]
        imgs = torch.stack([self.transform(img) for img in imgs])

        # breakpoint()

        unary_labels = [[a.contains_liquid, a.is_sealed, self.materials_enum[a.material], a.transparency] for a in acros]
         
        binary_labels = [self.deformability_matrix[pair[0]][pair[1]], self.fragility_matrix[pair[0]][pair[1]], self.mass_matrix[pair[0]][pair[1]]]

        label = [unary_labels, binary_labels]
        return imgs, [unary_labels, binary_labels]

    def pad_matrix(self, matrix, max_id):
        matrix = np.pad(matrix, ((0, max_id - matrix.shape[0]), (0, max_id - matrix.shape[1])), 'constant')
        return matrix

    def get_acro_to_id(self):
        mapping_acro_id = {}
        for i in self.acro_obj_list.keys():
            acro_id = self.acro_obj_list[i].acro_id
            mapping_acro_id[acro_id] = i
        return mapping_acro_id


    def get_pairs(self):

        deform_indices = np.nonzero(self.deformability_matrix)
        fragility_indices = np.nonzero(self.fragility_matrix)
        mass_indices = np.nonzero(self.mass_matrix)

        indices = np.concatenate((deform_indices, fragility_indices, mass_indices), axis=1).T  
        return indices

    
    def get_acro_from_fname(self, fname):
        pass
    
    def get_deformability_matrix(self):

        data = pd.read_csv(f"{self.phys_obs_base_dir}/deformability/{self.phys_obs_csv}.csv")

        id_dict = {}
        for i, row in data.iterrows():
            image_id_1 = int(row["annotation_id_0"])
            image_id_2 = int(row["annotation_id_1"])
            acro_id_1 = -1
            acro_id_2 = -1
            if image_id_1 not in self.acro_obj_list:
                self.acro_obj_list[image_id_1] = AcroObj(image_id_1, len(self.acro_obj_list))
            acro_id_1 = self.acro_obj_list[image_id_1].acro_id
            
            if image_id_2 not in self.acro_obj_list:
                self.acro_obj_list[image_id_2] = AcroObj(image_id_2, len(self.acro_obj_list))
            acro_id_2 = self.acro_obj_list[image_id_2].acro_id
            id_dict[(acro_id_1, acro_id_2)] = (row["response"]) 
        max_id = max(id_dict, key=lambda x: max(x))
        max_id = max(max_id[0], max_id[1])
        matrix = np.zeros((max_id + 1, max_id + 1))
        
        for key in id_dict.keys():
            vals = id_dict[key]
            if (vals == "left"):
                matrix[key[0]] [key[1]] = 1
                matrix[key[1]] [key[0]] = -1
            else:
                matrix[key[0]] [key[1]] = -1
                matrix[key[1]] [key[0]] = 1

        return matrix

    def get_fragility_matrix(self):
        
        data = pd.read_csv(f"{self.phys_obs_base_dir}/fragility/{self.phys_obs_csv}.csv")

        id_dict = {}
        for i, row in data.iterrows():
            image_id_1 = int(row["annotation_id_0"])
            image_id_2 = int(row["annotation_id_1"])
            acro_id_1 = -1
            acro_id_2 = -1
            if image_id_1 not in self.acro_obj_list:
                self.acro_obj_list[image_id_1] = AcroObj(image_id_1, len(self.acro_obj_list))
            acro_id_1 = self.acro_obj_list[image_id_1].acro_id
            
            if image_id_2 not in self.acro_obj_list:
                self.acro_obj_list[image_id_2] = AcroObj(image_id_2, len(self.acro_obj_list))
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

        return matrix

    def get_mass_matrix(self):
        
        data = pd.read_csv(f"{self.phys_obs_base_dir}/mass/{self.phys_obs_csv}.csv")

        id_dict = {}
        for i, row in data.iterrows():
            image_id_1 = int(row["annotation_id_0"])
            image_id_2 = int(row["annotation_id_1"])
            acro_id_1 = -1
            acro_id_2 = -1
            if image_id_1 not in self.acro_obj_list:
                self.acro_obj_list[image_id_1] = AcroObj(image_id_1, len(self.acro_obj_list))
            acro_id_1 = self.acro_obj_list[image_id_1].acro_id
            
            if image_id_2 not in self.acro_obj_list:
                self.acro_obj_list[image_id_2] = AcroObj(image_id_2, len(self.acro_obj_list))
            acro_id_2 = self.acro_obj_list[image_id_2].acro_id
            id_dict[(acro_id_1, acro_id_2)] = (row["response"]) 
        max_id = max(id_dict, key=lambda x: max(x))
        max_id = max(max_id[0], max_id[1])
        matrix = np.zeros((max_id + 1, max_id + 1))
        
        for key in id_dict.keys():
            vals = id_dict[key]
            if (vals == "left"):
                matrix[key[0]] [key[1]] = 1
                matrix[key[1]] [key[0]] = -1
            else:
                matrix[key[0]] [key[1]] = -1
                matrix[key[1]] [key[0]] = 1

        return matrix

    def read_contains_liquid(self):
        
        data = pd.read_csv(f"{self.phys_obs_base_dir}/can_contain_liquid/{self.phys_obs_csv}.csv")
        
        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                self.acro_obj_list[image_id].update({'contains_liquid': 1 if row["response"].lower() == "yes" else 0})
            else:
                self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), contains_liquid=1 if row["response"].lower() == "yes" else 0)
    
    def read_is_sealed(self):

        data = pd.read_csv(f"{self.phys_obs_base_dir}/is_sealed/{self.phys_obs_csv}.csv")

        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                self.acro_obj_list[image_id].update({'is_sealed': 1 if row["response"].lower() == "yes" else 0})
            else:
                self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), is_sealed=1 if row["response"].lower() == "yes" else 0)
    
    def material(self):
        materials = {}
        data = pd.read_csv(f"{self.phys_obs_base_dir}/material/{self.phys_obs_csv}.csv")

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
            else:
                if (materials[row["response"]] > 71 and row["response"] != "unknown"):
                    self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), material=row["response"])
        materials_copy = {key: val for key, val in materials.items() if val >= 71 and key != "unknown"}
        self.materials_enum = {key : index for index, key in enumerate(materials_copy)}
        self.materials_enum[-1] = -1

    def transparency(self):
        
        data = pd.read_csv(f"{self.phys_obs_base_dir}/transparency/{self.phys_obs_csv}.csv")

        for i, row in data.iterrows():
            image_id = int(row["annotation_id_0"])
            if image_id in self.acro_obj_list:
                self.acro_obj_list[image_id].update({'transparency': 1 if row["response"].lower() == "transparent" else 0})
            else:
                self.acro_obj_list[image_id] = AcroObj(image_id, len(self.acro_obj_list), transparency=1 if row["response"].lower() == "transparent" else 0)        
                
    def get_data(self):
        self.read_contains_liquid()
        self.read_is_sealed()
        self.material()
        self.transparency()
        self.get_deformability_matrix()
        self.get_fragility_matrix()
        self.get_mass_matrix()
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
    
    def read_phys_obs_data(self):
        with open('ego_objects_challenge_train.json', 'r') as file:
            json_data = json.load(file)
            
        #annotation ids
        self.data_info = self.get_data()
        all_ids = self.data_info.keys()
        ann = json_data.get("annotations", [])
        
        #image ids
        train_image_ids = []
        train_image_instance_ids = {}
        for entry in ann:
            if entry["id"] in all_ids:
                train_image_instance_ids[entry["image_id"]] = (entry["id"])
                train_image_ids.append(entry["image_id"])

        self.fname_to_train_id = {}
        filenames = []
        fname_to_acro = {}
        id_to_fname = {}
        for train_image_id in train_image_ids:
            image_entry = next((image for image in json_data["images"] if image["id"] == train_image_id), None)
            if image_entry:
                parts = image_entry["url"].split('/')
                new_url = '/' + '/'.join(parts[4:])
                file_path = os.path.basename(new_url)

                filenames.append(file_path)
                fname_to_acro[file_path] = self.data_info[train_image_instance_ids[train_image_id]]
                id_to_fname[self.data_info[train_image_instance_ids[train_image_id]].id] = file_path
                self.fname_to_train_id[file_path] = train_image_id

        return filenames, fname_to_acro, id_to_fname

        

if __name__ == "__main__":
    dataset = PhysObsDataset("") 
    inst = None
    inst = dataset.create_acro_dict()
    if inst is not None:
        print('done')