import torch
from data import data_loader
from vgg16_model import VGG16
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 20
batch_size = 8

model = VGG16(num_classes).to(device)
model_save_path = '/workspace/digital-wardrobe-recommendation/saved_models/vgg16_trained_model.pth'
model.load_state_dict(torch.load(model_save_path))

csv_file_path = './images.csv'
test_loader = data_loader(csv_file=csv_file_path, data_dir='/workspace/digital-wardrobe-recommendation', batch_size=batch_size, test=True)

model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
