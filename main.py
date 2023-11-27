from fastapi import FastAPI
import os
from typing import Dict
from predict import load_model, load_label_mappings, predict, get_num_classes, CustomDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(BASE_DIR, 'styles.csv')
model_path = os.path.join(BASE_DIR, 'saved_models', 'efficientnet_multioutput_model.pth')
label_mappings_path = os.path.join(BASE_DIR, 'label_mappings.json')

dataset = CustomDataset(csv_file=csv_file_path, data_dir='/workspace/digital-wardrobe-recommendation')
num_classes_dict = get_num_classes(dataset.data_frame)
model = load_model(model_path, num_classes_dict)
label_mappings = load_label_mappings(label_mappings_path)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Saketh Test!"}

@app.post("/predict/")
def make_prediction(input_data: Dict[str, str]):
    image_path = input_data.get("image_path")

    if not image_path:
        return {"error": "Please provide an 'image_path' field in the request body."}

    if not os.path.isfile(image_path):
        return {"error": "File does not exist. Please enter a valid path."}

    predictions = predict(image_path, model, num_classes_dict, label_mappings)
    return {"image_path": image_path, "predictions": predictions}