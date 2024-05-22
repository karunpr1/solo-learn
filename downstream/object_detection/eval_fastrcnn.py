import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
from collections import OrderedDict
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Custom ResNet50 class without the fully connected layer
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 2048  # Set the out_channels attribute

    def forward(self, x):
        return self.body(x)

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

# Paths to your data and annotations
annotation_file = '/home/karun/detectron2/datasets/price_labels/price_labels_valid/_annotations.coco.json'
image_dir = '/home/karun/detectron2/datasets/price_labels/price_labels_valid/images'  # Update this path to your actual image directory
  # Update this path to your actual image directory
model_path = '/home/karun/master_thesis/solo-learn/downstream/object_detection/output/test_plot_dino_pricetagmix_fasterrcnn_resnet50_10eps.pth'  # Path to the saved model

# Create a data loader for the validation set
val_dataset = CocoDataset(annotation_file, image_dir, transforms=T.ToTensor())
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_pred_scores = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, image in enumerate(images):
                true_labels = targets[i]['labels'].cpu().numpy().tolist()
                if 'labels' not in outputs[i] or len(outputs[i]['labels']) == 0:
                    pred_labels = []
                    pred_scores = []
                else:
                    pred_labels = outputs[i]['labels'].cpu().numpy().tolist()
                    pred_scores = outputs[i]['scores'].cpu().numpy().tolist()

                # Convert pred_scores to numpy array for comparison
                pred_scores = np.array(pred_scores)
                high_score_idx = pred_scores > 0.5
                pred_labels = np.array(pred_labels)[high_score_idx].tolist()
                pred_scores = pred_scores[high_score_idx].tolist()

                if len(pred_labels) == 0:
                    # Ensure at least one entry for compatibility with sklearn metrics
                    pred_labels = [-1]
                    pred_scores = [0]

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)
                all_pred_scores.extend(pred_scores)

                # Detailed Debugging Information
                print(f"Batch: {batch_idx}, Image Index: {i}, Image ID: {targets[i]['image_id'].item()}")
                print(f"True Labels: {true_labels}")
                print(f"Predicted Labels: {pred_labels}")
                print(f"Predicted Scores: {pred_scores}")

    # Ensure lengths are consistent
    min_length = min(len(all_true_labels), len(all_pred_labels), len(all_pred_scores))
    all_true_labels = all_true_labels[:min_length]
    all_pred_labels = all_pred_labels[:min_length]
    all_pred_scores = all_pred_scores[:min_length]

    # Further debugging information
    print(f"Final True Labels Length: {len(all_true_labels)}")
    print(f"Final Pred Labels Length: {len(all_pred_labels)}")
    print(f"Final Pred Scores Length: {len(all_pred_scores)}")

    print(f"All True Labels: {all_true_labels}")
    print(f"All Pred Labels: {all_pred_labels}")
    print(f"All Pred Scores: {all_pred_scores}")

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
    ap = average_precision_score(all_true_labels, all_pred_scores, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_precision": ap,
        "confusion_matrix": cm
    }



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
                visualize_prediction(image.cpu(), result)
    return results


# Load the model for inference
num_classes = 2  # 1 class (pricetag) + 1 background
backbone = ResNetBackbone()  # Instantiate the custom backbone
model = get_model(num_classes, backbone)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load(model_path))
model.to(device)

# Evaluate the model
results = evaluate_model(model, val_loader, device)
print("Evaluation Results:")
for metric, value in results.items():
    if metric == "confusion_matrix":
        print(f"{metric}:\n{value}")
    else:
        print(f"{metric}: {value:.4f}")

# Perform inference on the validation set
inference_results = inference(model, val_loader, device)
print("Inference completed.")
