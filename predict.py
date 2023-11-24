import torch
from torchvision import transforms
from PIL import Image
from data import CustomDataset, get_num_classes
from model import EfficientNetB0
import json

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
        decoded[key] = label_mappings[key]['idx_to_label'][str(value)]
    return decoded

def predict(image_path, model, num_classes_dict, label_mappings):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = transform_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        predictions = {col: torch.argmax(outputs[col], dim=1).item() for col in num_classes_dict}
    return decode_predictions(predictions, label_mappings)

# Main
if __name__ == "__main__":
    dataset = CustomDataset(csv_file='styles.csv', data_dir='/workspace/digital-wardrobe-recommendation')
    num_classes_dict = get_num_classes(dataset.data_frame)

    model_path = './saved_models/efficientnet_multioutput_model.pth'
    model = load_model(model_path, num_classes_dict)

    label_mappings = load_label_mappings('./label_mappings.json')

    image_path = './shirt.jpg'
    predictions = predict(image_path, model, num_classes_dict, label_mappings)
    print("Decoded Predictions:", predictions)
