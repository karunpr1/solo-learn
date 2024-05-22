from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import random

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def get_pricetag_dicts(img_dir):
    json_file = os.path.join(img_dir, "_annotations.coco.json")
    with open(json_file) as f:
        coco_dict = json.load(f)

    dataset_dicts = []
    for img_data in coco_dict['images']:
        record = {}

        filename = os.path.join(img_dir, 'images', img_data["file_name"])
        height, width = img_data["height"], img_data["width"]

        record["file_name"] = filename
        record["image_id"] = img_data["id"]
        record["height"] = height
        record["width"] = width

        annos = [anno for anno in coco_dict['annotations'] if anno['image_id'] == img_data['id']]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"] - 1,  # Adjust category_id to start from 0
                "segmentation": anno["segmentation"],
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Register test dataset
DatasetCatalog.register("price_labels_valid", lambda: get_pricetag_dicts("/home/karun/detectron2/datasets/price_labels/price_labels_valid"))
MetadataCatalog.get("price_labels_valid").set(thing_classes=["pricetag"])

# Load the trained model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # Path to the trained model weights
cfg.MODEL.WEIGHTS = "/home/karun/master_thesis/solo-learn/downstream/object_detection/fasterrcnn_resnet50.pth"  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90# Set a custom testing threshold
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.75
cfg.DATASETS.TEST = ("pricetag_valid")

# Create a predictor
predictor = DefaultPredictor(cfg)
#
# # Load test dataset
dataset_dicts = get_pricetag_dicts("/home/karun/detectron2/datasets/price_labels/price_labels_valid")

# Run inference and visualize results on a few test images
for d in random.sample(dataset_dicts, 10):  # Visualize a few samples randomly
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("pricetag_valid"), scale=0.9)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Predicted Image', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
cv2.destroyAllWindows()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Create evaluator
evaluator = COCOEvaluator("price_labels_valid", cfg, False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "price_labels_valid")

# Run evaluation
print(inference_on_dataset(predictor.model, test_loader, evaluator))