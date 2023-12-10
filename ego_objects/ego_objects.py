# Example usage to get annotations for specific images
annotation_path = 'path/to/test.json'  # Replace with the actual path
ego_objects = EgoObjects(annotation_path)

# Specify image filenames for which you want annotations
image_filenames_to_test = [
    '0A0DD98EB432BFD4563DAB1750D552FF_02_0.jpg',
    '0A0DD98EB432BFD4563DAB1750D552FF_02_01.jpg',
    # Add more image filenames as needed
]

# Get image IDs for the specified filenames
image_ids_to_test = [img["id"] for img in ego_objects.dataset["images"] if img["file_name"] in image_filenames_to_test]

# Get annotations for the specified images
annotation_ids = ego_objects.get_ann_ids(img_ids=image_ids_to_test)

# Load the annotations
annotations = ego_objects.load_anns(ids=annotation_ids)

# Now, 'annotations' contains the annotation information for the specified images
