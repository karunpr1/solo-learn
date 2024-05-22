import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Custom ResNet50 class without the fully connected layer
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 2048  # Set the out_channels attribute

    def forward(self, x):
        return self.body(x)


def load_resnet50_backbone_from_ckpt(ckpt_path):
    # Instantiate the custom ResNet50 backbone model
    backbone = ResNetBackbone()

    # Load weights from the .ckpt file
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']

    # Adjust keys for the backbone model
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    # Load the state_dict into the backbone model
    missing_keys, unexpected_keys = backbone.load_state_dict(new_state_dict, strict=False)

    # Print missing and unexpected keys, if any
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # Set all parameters to not require gradients
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone


class CocoDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.image_ids = [img['id'] for img in self.coco['images']]
        self.image_infos = {img['id']: img for img in self.coco['images']}
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_infos[image_id]
        image = Image.open(os.path.join(self.image_dir, image_info['file_name'])).convert("RGB")
        annotations = self.annotations[image_id]

        boxes = []
        labels = []
        for ann in annotations:
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)


def get_model(num_classes, backbone):
    # Use the provided backbone and add an FPN to it
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        ),
        box_roi_pool=torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
    )
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# Example usage to load the resnet50 backbone from the checkpoint file
ckpt_path = "/home/karun/master_thesis/solo-learn/trained_models/dino/yze579eg/dino-pricetag_mix-dataset-yze579eg-ep=799.ckpt" # Update this path to your actual checkpoint path
resnet50_backbone = load_resnet50_backbone_from_ckpt(ckpt_path)

# Paths to your data and annotations
annotation_file = '/home/karun/detectron2/datasets/price_labels/price_labels_train/_annotations.coco.json'
image_dir = '/home/karun/detectron2/datasets/price_labels/price_labels_train/images'  # Update this path to your actual image directory

# Define the dataset and data loader
dataset = CocoDataset(annotation_file, image_dir, transforms=T.ToTensor())
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)

val_dir = "/home/karun/detectron2/datasets/price_labels/price_labels_valid/images"
val_annotations = "/home/karun/detectron2/datasets/price_labels/price_labels_valid/_annotations.coco.json"

# Create a data loader for the validation set
val_dataset = CocoDataset(val_annotations, val_dir, transforms=T.ToTensor())
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)

# Model
num_classes = 2  # 1 class (pricetag) + 1 background
model = get_model(num_classes, resnet50_backbone)

# Training parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 100
train_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    i = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        i += 1
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(data_loader)}]: Loss: {losses.item():.4f}")

    avg_epoch_loss = epoch_loss / len(data_loader)
    train_losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_epoch_loss:.2f}")

# Save the training losses to a CSV file
loss_df = pd.DataFrame(train_losses, columns=['loss'])
loss_df.to_csv('pricetagmix_100eps_dino_training_losses.csv', index=False)
print("Training losses saved to training_losses.csv")

# Plot the epoch vs loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.grid()
plt.savefig('pricetagmix_100eps_dino_training_losses.png')
plt.show()
print("Loss graph saved to epoch_vs_loss.png")

print("Training completed.")

model_name = '/home/karun/master_thesis/solo-learn/downstream/object_detection/output/dino_fastrcnn_resnet50_pricetagmix_100eps.pth'

# Save the trained model
torch.save(model.state_dict(), model_name)
print("Model saved.")

# Inference on the validation set
def inference(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, image in enumerate(images):
                result = {
                    'image_id': targets[i]['image_id'].item(),
                    'boxes': outputs[i]['boxes'].cpu().numpy(),
                    'labels': outputs[i]['labels'].cpu().numpy(),
                    'scores': outputs[i]['scores'].cpu().numpy()
                }
                results.append(result)
                # Optionally, visualize the predictions
                visualize_prediction(image.cpu(), result)
    return results

def visualize_prediction(image, result):
    image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
        if score > 0.5:  # Filter out low-score predictions
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                       fill=False, edgecolor='red', linewidth=2))
            ax.text(box[0], box[1], f'price_tag: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()


# # Load the model for inference
# model.load_state_dict(torch.load(model_name))
# model.to(device)
#
# # Perform inference on the validation set
# results = inference(model, val_loader, device)
# print("Inference completed.")