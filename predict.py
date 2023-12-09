import torch
from torchvision import transforms
from PIL import Image
from data import CustomDataset, get_num_classes
from model import EfficientNetB0
import json
import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(model_path, num_classes_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetB0(num_classes_dict).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def transform_image(image_path):
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


def load_label_mappings(mapping_file_path):
    with open(mapping_file_path, 'r') as file:
        return json.load(file)


def decode_predictions(predictions, label_mappings):
    decoded = {}
    for key, value in predictions.items():
        if key != 'season':
            decoded[key] = label_mappings[key]['idx_to_label'][str(value)]
    return decoded


def predict(image_path, model, num_classes_dict, label_mappings):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = transform_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        predictions = {col: torch.argmax(outputs[col], dim=1).item() for col in num_classes_dict}
    return decode_predictions(predictions, label_mappings)

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.image_path):
        print("File does not exist. Please enter a valid path.")
    else:
        csv_file_path = os.path.join(BASE_DIR, 'styles.csv')
        model_path = os.path.join(BASE_DIR, 'saved_models', 'efficientnet_multioutput_model.pth')
        label_mappings_path = os.path.join(BASE_DIR, 'label_mappings.json')

        dataset = CustomDataset(csv_file=csv_file_path, data_dir='/workspace/digital-wardrobe-recommendation')
        num_classes_dict = get_num_classes(dataset.data_frame)

        model = load_model(model_path, num_classes_dict)

        label_mappings = load_label_mappings(label_mappings_path)

        predictions = predict(args.image_path, model, num_classes_dict, label_mappings)
        print("Decoded Predictions:", predictions)