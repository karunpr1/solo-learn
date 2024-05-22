import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.roi_heads import FastRCNNOutputLayers

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
                "category_id": anno["category_id"] - 1,
                "segmentation": anno["segmentation"],
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# Register datasets
for d in ["train", "valid"]:
    DatasetCatalog.register("pricetag_" + d, lambda d=d: get_pricetag_dicts(f"/home/karun/detectron2/datasets/price_labels/price_labels_{d}"))
    MetadataCatalog.get("pricetag_" + d).set(thing_classes=["pricetag"])

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

# Configuration setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("pricetag_train",)
cfg.DATASETS.TEST = ("pricetag_valid",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "/home/karun/master_thesis/solo-learn/downstream/object_detection/dino_imagenet_pretrained.pkl"  # Path to the pre-trained BYOL model
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.SOLVER.STEPS = (7000, 9000)
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
cfg.MODEL.RETINANET.NUM_CLASSES = 1  # Number of object classes
cfg.MODEL.RETINANET.NMS_THRESH_TRAIN = 0.6
cfg.MODEL.BACKBONE.FREEZE_AT = 0

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Verify dataset registration
assert "pricetag_train" in DatasetCatalog.list(), "pricetag_train not registered!"
assert "pricetag_valid" in DatasetCatalog.list(), "pricetag_valid not registered!"

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
