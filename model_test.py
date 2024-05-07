import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from solo.methods.linear import LinearModel  # imports the linear eval class
# from solo.utils.classification_dataloader import prepare_data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),               # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),           # Crop the center 224x224 pixels
    transforms.ToTensor(),                # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pretrained ResNet
])
test_transform = transform

train_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset/train'
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)


test_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset/val'
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = resnet50(pretrained=True)
num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, 25)  # Adjust for 25 classes
state = torch.load('/home/karun/master_thesis/solo-learn/trained_models/dino/poassag1/dino-custom-dataset-poassag1-ep=299.ckpt')['state_dict']
backbone.load_state_dict(state, strict=False)


num_features = 2048  # Get the input features of the original fc layer
backbone.fc = nn.Linear(num_features, 25)  # Replace it with a new one for 25 classes

# Freeze all layers except the final fully connected
for param in backbone.parameters():
    param.requires_grad = False
backbone.fc.weight.requires_grad = True
backbone.fc.bias.requires_grad = True  # Only train the last layer
backbone = backbone.to(device)
# Define your optimizer and criterion
optimizer = torch.optim.Adam(backbone.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# train_model(backbone, criterion, optimizer, train_loader, num_epochs=100)

# torch.save(backbone.state_dict(), '/home/karun/saved_models/trained_resnet50.pth')


def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy

# Load the model for evaluation
# backbone = resnet50(pretrained=True)
backbone.fc = nn.Linear(num_features, 25)  # Adjust again for 25 classes
backbone.load_state_dict(torch.load('/home/karun/saved_models/trained_resnet50.pth'))
backbone = backbone.to(device)

# Evaluate the model
accuracy = evaluate_model(backbone, test_loader, device)
# print(f'Accuracy of the model on the test images: {accuracy:.2f}%')