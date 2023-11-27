import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import json

class LabelEncoder:
    def __init__(self):
        self.label_to_idx = {}
        self.idx_to_label = {}

    def fit(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def encode(self, label):
        return self.label_to_idx.get(label, -1)

    def decode(self, idx):
        return self.idx_to_label.get(idx, 'Unknown')

    @staticmethod
    def save_all_mappings(encoders, file_path):
        all_mappings = {col: {'label_to_idx': encoder.label_to_idx, 'idx_to_label': encoder.idx_to_label}
                        for col, encoder in encoders.items()}
        with open(file_path, 'w') as file:
            json.dump(all_mappings, file, indent=4)

    @staticmethod
    def load_all_mappings(file_path):
        with open(file_path, 'r') as file:
            all_mappings = json.load(file)
        encoders = {col: LabelEncoder() for col in all_mappings}
        for col, mappings in all_mappings.items():
            encoders[col].label_to_idx = mappings['label_to_idx']
            encoders[col].idx_to_label = mappings['idx_to_label']
        return encoders

class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, save_mappings=False):
        self.data_frame = pd.read_csv(csv_file, on_bad_lines='skip')
        self.data_dir = data_dir
        self.transform = transform

        for col in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']:
            self.data_frame[col] = self.data_frame[col].astype(str)

        self.data_frame = self.data_frame.astype(object).fillna('Unknown')

        self.label_encoders = {col: LabelEncoder() for col in
                               ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season',
                                'usage']}

        for col, encoder in self.label_encoders.items():
            encoder.fit(self.data_frame[col])

        if save_mappings:
            LabelEncoder.save_all_mappings(self.label_encoders, './label_mappings.json')


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = self.data_frame.iloc[idx, 0]
        img_name = os.path.join(self.data_dir, 'images', f"{img_id}.jpg")
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = {col: torch.tensor(self.label_encoders[col].encode(self.data_frame.iloc[idx][col]), dtype=torch.long)
                  for col in self.label_encoders}

        return image, labels

def get_num_classes(data_frame):
    num_classes_dict = {}
    for col in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']:
        num_classes = data_frame[col].nunique()
        num_classes_dict[col] = num_classes
    return num_classes_dict

def data_loader(csv_file,
                data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        test_dataset = CustomDataset(csv_file=csv_file, data_dir=data_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader

    full_dataset = CustomDataset(csv_file=csv_file, data_dir=data_dir, transform=transform)

    num_train = len(full_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader
