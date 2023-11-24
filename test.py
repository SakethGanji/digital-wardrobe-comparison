import torch
from data import data_loader, get_num_classes, CustomDataset
from model import EfficientNetB0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

dataset = CustomDataset(csv_file='styles.csv', data_dir='/workspace/digital-wardrobe-recommendation', save_mappings=True)
num_classes_dict = get_num_classes(dataset.data_frame)

model = EfficientNetB0(num_classes_dict).to(device)
model_save_path = '/workspace/digital-wardrobe-recommendation/saved_models/efficientnet_multioutput_model.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))

csv_file_path = './styles.csv'
test_loader = data_loader(csv_file=csv_file_path, data_dir='/workspace/digital-wardrobe-recommendation', batch_size=batch_size, test=True)

model.eval()

metrics = {col: {'true_labels': [], 'predicted_labels': []} for col in num_classes_dict}

with torch.no_grad():
    for images, labels_dict in test_loader:
        images = images.to(device)
        outputs_dict = model(images)

        for col in num_classes_dict:
            _, predicted = torch.max(outputs_dict[col].data, 1)
            metrics[col]['true_labels'].extend(labels_dict[col].cpu().numpy())
            metrics[col]['predicted_labels'].extend(predicted.cpu().numpy())

for col in num_classes_dict:
    true_labels = np.array(metrics[col]['true_labels'])
    predicted_labels = np.array(metrics[col]['predicted_labels'])

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    print(f"Metrics for {col}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}\n")
