import os
import re
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def extract_id_from_filename(filename):
    match = re.search(r'(\d+)_', os.path.basename(filename))
    return int(match.group(1)) if match else None

def load_folder_dict(folder_path):
    files = glob(os.path.join(folder_path, '*.npy'))
    data_dict = {}

    for file in files:
        file_id = extract_id_from_filename(file)
        if file_id is None:
            continue
        data_dict[file_id] = np.load(file)
    
    return data_dict

resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BrainTumorDataset(Dataset):
    def __init__(self, image_dict, label_dict, ids, transform=None):
        self.image_dict = image_dict
        self.label_dict = label_dict
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        file_id = self.ids[idx]

        image = self.image_dict[file_id]
        label = self.label_dict[file_id]

        image = image.astype(np.float32)
        image = image / 511.0

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if self.transform:
            image = image.transpose(1, 2, 0) # transpose for npy->PIL format 
            image = self.transform(image)

        label = torch.tensor(int(label.item())-1, dtype=torch.long) # 0 based index 

        return image, label
    
def create_datasets(data_dir, test_size=0.2, val_size=0.1,random_state=42, transform=None):
    image_folder = os.path.join(data_dir, 'images')
    label_folder = os.path.join(data_dir, 'labels')

    print("Loading images from folder: ", image_folder)
    image_dict = load_folder_dict(image_folder)
    print("Loading labels from folder: ", label_folder)
    label_dict = load_folder_dict(label_folder)

    ids = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))

    if len(ids) == 0:
        raise ValueError("No matching IDs found between images and labels.")
    
    print(f"Total samples found: {len(ids)}")

    train_val_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, random_state=random_state)

    print(f"Training samples: {len(train_ids)}, Validation samples: {len(val_ids)}, Testing samples: {len(test_ids)}")

    train_dataset = BrainTumorDataset(image_dict, label_dict, train_ids, transform=resnet_transform)
    val_dataset = BrainTumorDataset(image_dict, label_dict, val_ids, transform=resnet_transform)
    test_dataset = BrainTumorDataset(image_dict, label_dict, test_ids, transform=resnet_transform)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    data_dir = "data"

    train_dataset, val_dataset, test_dataset = create_datasets(data_dir)

    print("Sanity check:")
    image, label = train_dataset[0]

    print("Image shape:", image.shape)
    print("label:", label)