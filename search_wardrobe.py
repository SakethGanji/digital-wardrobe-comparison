import faiss
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import joblib
import os

from data import CustomDataset, get_num_classes
from model import EfficientNetB0, load_model

dataset = CustomDataset(csv_file='styles.csv', data_dir='/workspace/digital-wardrobe-recommendation', save_mappings=True)
num_classes_dict = get_num_classes(dataset.data_frame)

path = "./saved_models/efficientnet_multioutput_model.pth"
model = load_model(path, num_classes_dict)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

pca = joblib.load('pca_model.joblib')
faiss_index = faiss.read_index('wardrobe_index.index')

modified_wardrobe_dir = './modified_wardrobe'
wardrobe_filenames = [f for f in os.listdir(modified_wardrobe_dir) if f.endswith('.jpg')]

def process_image(image_path):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def search_wardrobe(image_path, k=5, threshold=None):
    image = process_image(image_path)
    image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        feature = model.get_feature_vector(image).cpu().numpy()

    reduced_feature = pca.transform(feature)
    faiss.normalize_L2(reduced_feature)

    distances, indices = faiss_index.search(reduced_feature.astype('float32'), k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if threshold is None or distance < threshold:
            match = {
                'filename': wardrobe_filenames[idx],
                'distance': distance
            }
            results.append(match)

    return results
