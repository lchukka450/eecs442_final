'''
import json

class EgoObjects:
    def __init__(self, annotation_path, annotation_dict=None):
        # Simplified implementation for demonstration purposes
        # You should replace this with your actual EgoObjects class

        self.dataset = self._load_json(annotation_path)
        self._create_index()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        # Simplified index creation
        self.img_ann_map = {ann["image_id"]: ann for ann in self.dataset["annotations"]}

    def get_ann_ids(self, img_ids=None):
        # Simplified method to get annotation IDs
        if img_ids is not None:
            return [self.img_ann_map[img_id]["id"] for img_id in img_ids]
        else:
            return [ann["id"] for ann in self.dataset["annotations"]]

    def load_anns(self, ids=None):
        # Simplified method to load annotations
        if ids is not None:
            return [self.img_ann_map[img_id] for img_id in ids]
        else:
            return self.dataset["annotations"]
        
    def get_annotations_for_images(self, image_filenames):
        # Get image IDs for the specified filenames
        image_ids_to_test = [img["id"] for img in self.dataset["images"] if img["id"] in image_filenames]

        # Get annotation IDs for the specified image IDs
        annotation_ids_for_images = self.get_ann_ids(image_ids_to_test)

        # Load annotations for the specified annotation IDs
        annotations_for_images = self.load_anns(annotation_ids_for_images)

        return annotations_for_images


def main():
    annotation_path = 'ego_objects_challenge_test.json'  # Replace with the actual path
    image_filenames_to_test = [
        '0A0DD98EB432BFD4563DAB1750D552FF_02_0.jpg',
        '0A0DD98EB432BFD4563DAB1750D552FF_02_01.jpg',
        # Add more image filenames as needed
    ]

    # Load annotations for specified images
    ego_objects = EgoObjects(annotation_path)
    
    # Print keys in the "images" part of the dataset
    print("Keys in the 'images' part of the dataset:", ego_objects.dataset["images"][0].keys())

    # Uncomment the line below once you identify the correct key for image filenames
    # image_ids_to_test = [img["id"] for img in ego_objects.dataset["images"] if img["<correct_key>"] in image_filenames_to_test]

    # Print the loaded annotations
    # print(json.dumps(annotations, indent=2))
    annotations_for_images = ego_objects.get_annotations_for_images(image_filenames_to_test)
    print(json.dumps(annotations_for_images, indent=2))

if __name__ == "__main__":
    main()
    
'''
'''
import json
import os

# Load the JSON file
json_file_path = "ego_objects_challenge_test.json"
with open(json_file_path, "r") as f:
    data = json.load(f)

# Specify the path to the "images" folder
images_folder_path = "images"

# Get the list of images in the folder
image_filenames = [filename for filename in os.listdir(images_folder_path) if filename.endswith(".jpg")]

# Iterate through each image in the folder
for image_filename in image_filenames:
    # Find the corresponding entry in the JSON data
    matching_entries = [entry for entry in data["images"] if entry["manifold://ego_objects_v1/tree/images_and_annotations/images/0A0DD98EB432BFD4563DAB1750D552FF_02_1.jpg"].endswith(image_filename)]
    
    # Check if there is a match
    if matching_entries:
        # Extract the main category label from the matching entry
        main_category = matching_entries[0]["main_category"]
        print(f"Image: {image_filename}, Main Category: {main_category}")
    else:
        print(f"No match found for {image_filename}")
'''

import json

# Load your JSON file
with open('ego_objects_challenge_test.json', 'r') as file:
    data = json.load(file)

# Sample image IDs for which you want annotations
sample_image_ids = [667431057639087, 399487875311908, 1869641053243174, 981235229161083, 329001905834822]

# Extracting annotations
annotations = data.get("annotations", [])

# Print annotations for the sample images
print("\nAnnotations for Sample Images:")
for annotation in annotations:
    if annotation.get("image_id") in sample_image_ids:
        print("Annotation ID:", annotation.get("id"))
        print("Image ID:", annotation.get("image_id"))
        print("Category ID:", annotation.get("category_id"))
        print("------")
        
for sample_image_id in sample_image_ids:
    # Find the image entry with the specified ID
    image_entry = next((image for image in data["images"] if image["id"] == sample_image_id), None)

    # Check if the image entry was found
    if image_entry:
        # Access the main_category value
        main_category = image_entry["main_category"]
        print(f"The main_category for image {sample_image_id} is: {main_category}")
        
        main_category_instance_ids = image_entry["main_category_instance_ids"]
        print(f"The main_category_instance_ids for image {sample_image_id} is: {main_category_instance_ids}")
        
        group_id = image_entry["group_id"]
        print(f"The group_id for image {sample_image_id} is: {group_id}")
    else:
        print(f"Image with ID {sample_image_id} not found.")

