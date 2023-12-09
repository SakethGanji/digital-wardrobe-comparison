import faiss
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from data import CustomDataset, get_num_classes
from model import load_model
import joblib
from rembg import remove
import pyheif

def remove_background(input_path, output_path):
    input_image = Image.open(input_path)
    output_image = remove(input_image)

    if output_image.mode == 'RGBA':
        output_image = output_image.convert('RGB')

    output_image.save(output_path)

def read_heic_image(image_path):
    heif_file = pyheif.read(image_path)
    return Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

def convert_to_jpg(input_path, output_path):
    if input_path.lower().endswith('.heic'):
        input_image = read_heic_image(input_path)
    else:
        input_image = Image.open(input_path)
    input_image.convert('RGB').save(output_path, 'JPEG')


image_directory = "./wardrobe"
jpg_directory = "./wardrobe_jpg"
os.makedirs(jpg_directory, exist_ok=True)

for img_file in os.listdir(image_directory):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
        original_path = os.path.join(image_directory, img_file)
        jpg_path = os.path.join(jpg_directory, os.path.splitext(img_file)[0] + '.jpg')
        convert_to_jpg(original_path, jpg_path)

image_directory = jpg_directory
modified_image_directory = "./modified_wardrobe"
os.makedirs(modified_image_directory, exist_ok=True)

for img_file in os.listdir(image_directory):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        original_path = os.path.join(image_directory, img_file)
        modified_path = os.path.join(modified_image_directory, img_file)
        remove_background(original_path, modified_path)

dataset = CustomDataset(csv_file='styles.csv', data_dir='/workspace/digital-wardrobe-recommendation', save_mappings=True)
num_classes_dict = get_num_classes(dataset.data_frame)
path = "./saved_models/efficientnet_multioutput_model.pth"
model = load_model(path, num_classes_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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

index = faiss.IndexFlatL2(128)

features_list = []
for img_file in os.listdir(modified_image_directory):
    if img_file.endswith('.jpg'):
        img_path = os.path.join(modified_image_directory, img_file)
        image = process_image(img_path)
        image = image.to(device)
        with torch.no_grad():
            feature = model.get_feature_vector(image).cpu().numpy()
        features_list.append(feature[0])

features_array = np.array(features_list)

num_samples, num_features = features_array.shape
n_components = min(num_samples, num_features, 128)

pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(features_array)
reduced_features = np.ascontiguousarray(reduced_features)

faiss.normalize_L2(reduced_features)

index = faiss.IndexFlatL2(n_components)

index.add(reduced_features)

faiss.write_index(index, 'wardrobe_index.index')
np.save('pca_components.npy', pca.components_)
joblib.dump(pca, 'pca_model.joblib')
