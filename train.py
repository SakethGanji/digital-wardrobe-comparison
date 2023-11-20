import os
import torch
import torch.nn as nn
from vgg16_model import VGG16
from data import data_loader

num_classes = 20
num_epochs = 40
batch_size = 8
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGG16(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

csv_file_path = './images.csv'
train_loader, valid_loader = data_loader(
    csv_file='images.csv',
    data_dir='/workspace/digital-wardrobe-recommendation',
    batch_size=batch_size
)

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}]')

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model_save_path = '/workspace/digital-wardrobe-recommendation/saved_models/vgg16_trained_model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
