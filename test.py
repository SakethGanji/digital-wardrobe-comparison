import torch
from data import data_loader
from model import EfficientNetB0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 143
batch_size = 8

model = EfficientNetB0(num_classes).to(device)
model_save_path = '/workspace/digital-wardrobe-recommendation/saved_models/efficientnet_trained_model.pth'
model.load_state_dict(torch.load(model_save_path, map_location=device))

csv_file_path = './styles.csv'
test_loader = data_loader(csv_file=csv_file_path, data_dir='/workspace/digital-wardrobe-recommendation', batch_size=batch_size, test=True)

model.eval()

true_labels = []
predicted_labels = []
all_outputs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())

# Convert to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
all_outputs = np.array(all_outputs)

# Metrics calculation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
