import json

import torch
from torchvision import transforms
from PIL import Image

from vgg16_model import VGG16
import pandas as pd

model_save_path = '/workspace/digital-wardrobe-recommendation/saved_models/vgg16_trained_model.pth'
num_classes = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('./images.csv')
all_labels = df['label'].unique()


def load_trained_model(model_path):
    model = VGG16(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def predict(image_path, model):
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()


def load_label_mapping(file_path):
    with open(file_path, 'r') as file:
        idx_to_label = json.load(file)
    return {int(k): v for k, v in idx_to_label.items()}


def decode_prediction(prediction_index, mapping):
    return mapping.get(prediction_index, "Unknown")


trained_model = load_trained_model(model_save_path)

label_mapping = load_label_mapping('label_mapping.json')
new_image_path = './shirt.jpg'
prediction = predict(new_image_path, trained_model)
predicted_label = decode_prediction(prediction, label_mapping)
print("Predicted class:", predicted_label)
