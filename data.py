import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import json

class LabelEncoder:
    def __init__(self, labels):
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def encode(self, label):
        return self.label_to_idx[label]

    def decode(self, idx):
        return self.idx_to_label[idx]

    def save_mapping(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.idx_to_label, file, indent=4)

class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file, on_bad_lines='skip')
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = LabelEncoder(self.data_frame['articleType'])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = self.data_frame.iloc[idx, 0]
        img_name = os.path.join(self.data_dir, 'images', f"{img_id}.jpg")
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_str = self.data_frame.iloc[idx, 4]
        label_int = self.label_encoder.encode(label_str)
        self.label_encoder.save_mapping('label_mapping.json')
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return image, label_tensor


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

