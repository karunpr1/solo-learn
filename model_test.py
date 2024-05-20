import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torchvision.models import resnet18, resnet50, ResNet50_Weights
import torch.nn as nn
from solo.methods.linear import LinearModel  # imports the linear eval class
# from solo.utils.classification_dataloader import prepare_data
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),  # Crop the center 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pretrained ResNet
])
test_transform = transform

# 100% train data
# train_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset/train'
# 50% train data
# train_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset_50_perc/train'
# 30% train data
# train_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset_30_perc/train'
# 10% train data
train_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset_10_perc/train'

train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

test_data_dir = '/home/karun/master_thesis/datasets/imagenet_like/freiburg_groceries_dataset/val'
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = resnet50(pretrained=False)
# backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in backbone.parameters():
    param.requires_grad = False

num_features = backbone.fc.in_features
num_classes = 25
backbone.fc = nn.Linear(num_features, num_classes)
mlp_layer = nn.Sequential(
               nn.Linear(2048, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, num_classes))  # Adjust for 25 classes
state = torch.load('/home/karun/master_thesis/solo-learn/trained_models/dino/kidphni2/dino-custom-dataset-kidphni2-ep=399.ckpt')['state_dict']
backbone.load_state_dict(state, strict=False)


# Freeze all layers except the final fully connected
backbone = backbone.to(device)
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
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


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

#
# train_model(backbone, criterion, optimizer, train_loader, num_epochs=100)
# torch.save(backbone.state_dict(), '/home/karun/saved_models/mlp_linear/dummy_test/dino_trained_resnet50_linear_100eps_10perc.pt')


# Evaluate the model
# print("Dino Pre-trained")
# backbone.load_state_dict(torch.load('/home/karun/saved_models/mlp_linear/dummy_test/dino_trained_resnet50_linear_100eps_10perc.pt'))
# evaluate_model(backbone, test_loader, device)


# print("Imagenet Pre-trained")
# backbone.load_state_dict(torch.load('/home/karun/saved_models/mlp_linear/imagenet_trained_resnet50_linear_100eps_10perc.pt'))
# accuracy = evaluate_model(backbone, test_loader, device)


# print("Random weights Pre-trained")
# backbone.load_state_dict(torch.load('/home/karun/saved_models/random_weights_resnet50_linear_100eps_10perc.pth'))
# accuracy = evaluate_model(backbone, test_loader, device)

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet50
from torchvision import transforms
from PIL import Image, ImageDraw
import os


# Load the pre-trained ResNet50 backbone
def load_backbone(backbone_path):
    backbone = resnet50(pretrained=False)
    checkpoint = torch.load(backbone_path)
    state_dict = checkpoint.get('model', checkpoint)
    backbone.load_state_dict(
        {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}, strict=False)

    # Remove the fully connected layer and the avgpool layer
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    # Add the out_channels attribute
    backbone.out_channels = 2048

    return backbone


# Construct the Faster R-CNN model
def construct_fasterrcnn_model(backbone, num_classes=2):
    # Create the anchor generator for the FPN
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Create the RoIAlign pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    # Create the Faster R-CNN model using the backbone
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


# Load the complete model
def load_model(model_path, backbone_path, num_classes=2):
    backbone = load_backbone(backbone_path)
    model = construct_fasterrcnn_model(backbone, num_classes)

    # Load the state_dict for the full model, excluding roi_heads.box_predictor layers
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('model', checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if 'roi_heads.box_predictor' not in k}
    model.load_state_dict(state_dict, strict=False)

    # Reinitialize the box_predictor to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.eval()
    return model


# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


# Perform inference on an image
def detect_price_labels(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction


# Display the predictions
def display_predictions(image_path, predictions, threshold=0.85):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(predictions[0]['boxes']):
        if predictions[0]['scores'][i] > threshold:
            draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)

    image.show()


# Main function to process a dataset
def process_dataset(model, dataset_dir):
    for image_filename in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_filename)

        # Preprocess the image
        image_tensor = preprocess_image(image_path)

        # Perform inference
        predictions = detect_price_labels(model, image_tensor)

        # Display the predictions
        display_predictions(image_path, predictions)


# Paths to the model and backbone
# model_path = '/home/karun/master_thesis/solo-learn/downstream/object_detection/output/model_final.pth'
# # print(load_model(model_path))
# backbone_path = '/home/karun/master_thesis/solo-learn/downstream/object_detection/output/model_final.pth'
# num_classes = 2  # Adjust this based on your dataset
#
# # Load the model
# model = load_model(model_path, backbone_path, num_classes)
# print(model)
#
# # Path to the dataset directory
# dataset_dir = '/home/karun/detectron2/datasets/price_labels/price_labels_test'
#
# # Process the dataset
# process_dataset(model, dataset_dir)


import torch
import torchvision.models.detection as detection
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define the model architecture
model = detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)

# Load the trained model's state_dict
model_path = "/home/karun/master_thesis/solo-learn/downstream/object_detection/output/model_final.pth"
checkpoint = torch.load(model_path)

# Load the state dict into the model, ignoring the ROI heads
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

# Filter out the keys related to roi_heads.box_predictor
backbone_keys = {k: v for k, v in state_dict.items() if not k.startswith('roi_heads.box_predictor')}

# Load the filtered state dict into the model
model.load_state_dict(backbone_keys, strict=False)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the head with a new one (note: +1 for background)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# Set the model to evaluation mode
model.eval()

# Example input image
image_path = "/home/karun/detectron2/datasets/price_labels/price_labels_test/264_jpg.rf.57abdb0ff860e3ae594a268a545dae04.jpg"
image = Image.open(image_path)
transform = T.Compose([T.ToTensor()])
input_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    outputs = model(input_tensor)

# Extract boxes and scores
boxes = outputs[0]['boxes']
scores = outputs[0]['scores']

# Threshold to filter boxes
threshold = 0.55
filtered_boxes = boxes[scores > threshold]

# Convert tensor to PIL image
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Draw bounding boxes
for box in filtered_boxes:
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

# Display the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
# print(outputs)


