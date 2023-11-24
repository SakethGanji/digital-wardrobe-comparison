import os
import torch
import torch.nn as nn
from model import EfficientNetB0
from data import data_loader, CustomDataset, get_num_classes

dataset = CustomDataset(csv_file='styles.csv', data_dir='/workspace/digital-wardrobe-recommendation', save_mappings=True)
num_classes_dict = get_num_classes(dataset.data_frame)

num_epochs = 40
batch_size = 32
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EfficientNetB0(num_classes_dict).to(device)

criterion_dict = {col: nn.CrossEntropyLoss() for col in num_classes_dict}

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

train_loader, valid_loader = data_loader(
    csv_file='styles.csv',
    data_dir='/workspace/digital-wardrobe-recommendation',
    batch_size=batch_size
)

for epoch in range(num_epochs):
    for i, (images, labels_dict) in enumerate(train_loader):
        images = images.to(device)
        labels_dict = {col: labels.to(device) for col, labels in labels_dict.items()}
        outputs_dict = model(images)

        loss = sum(criterion_dict[col](outputs_dict[col], labels_dict[col]) for col in num_classes_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

model_save_path = '/workspace/digital-wardrobe-recommendation/saved_models/efficientnet_multioutput_model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
