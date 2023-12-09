import torch
# import pd
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image
import json

instance_id_map = {}

def create_instance_id_mapping(fname):
    instance_id_map = {} # from ego objects challenge test & train
    
    with open(fname, 'r') as json_file:
        data = json.load(json_file)
        for item in data["annotations"]:
            instance_id = item.get('instance_id')
            if instance_id not in instance_id_map:
                instance_id_map[instance_id] = {'id': item.get('id'), 'category_id': item.get('category_id')}
    
    return instance_id_map

def get_info_ego(id): 
    return (instance_id_map[id])

def load_images_from_directory(directory):
    images = []
    count = 0 
    while (count < 10):
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg')):  # Add other image formats if needed
                img = Image.open(os.path.join(directory, filename))
                images.append(img)
                print(get_info_ego(filename))
                count+= 1
    return images

def get_training_ids(filepath):
    with open("physobjects/instance_ids/train_ids.json", 'r') as json_file:
        data = json.load(json_file)
        return data



def get_test_ids(filepath):
    with open("physobjects/instance_ids/test_ids.json", 'r') as json_file:
        data = json.load(json_file)
        return data 

def open_image(id):
    try:
        path = f"images/{id}.jpg"
        print(path)
        img = Image.open(path)
        img.show()  
    except FileNotFoundError:
        print("File not found. Please provide a valid filename.")
    except Exception as e:
        print(f"An error occurred: {e}")
    



# def load_attributes_from_csv(filepath):
#     try:
#         df = pd.read_csv(filepath)





def get_dataloader():
    images = load_images_from_directory()
    attributes = load_attributes_from_csv()
    custom_dataset = AttributeSet(images, attributes)
    batch_size = 32
    data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False)
    return data_loader


class PhysObsDataset(Dataset):
    def __init__(self, id_json_file):
        instance_ids = json.load(id_json_file)
        self.instance_to_id = 

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        attribute = self.attributes[idx]
        return image, attribute



# # load_images_from_directory("images/")
# print(get_training_ids(""))
if __name__ == '__main__':
    # 0A0DD98EB432BFD4563DAB1750D552FF_01_38    first image id 
    create_instance_id_mapping()
     #FD9994FA1EF18DCB71461E469635EE95_09_37
    print(get_info_ego("FD9994FA1EF18DCB71461E469635EE95_09_37"))
    # open_image("FD9994FA1EF18DCB71461E469635EE95_402_0")