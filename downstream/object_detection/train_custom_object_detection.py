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
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.layers import get_norm
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


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
train_img_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/"  # Assuming the images are in the 'images' folder inside this directory
DatasetCatalog.register("russian_price_labels_train", lambda: get_pricetag_dicts(train_img_dir))
MetadataCatalog.get("russian_price_labels_train").set(thing_classes=["-", "name", "old_price", "price", "promotion"])

test_img_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/"  # Assuming the images are in the 'images' folder inside this directory
DatasetCatalog.register("russian_price_labels_test", lambda: get_pricetag_dicts(train_img_dir))
MetadataCatalog.get("russian_price_labels_test").set(thing_classes=["-", "name", "old_price", "price", "promotion"])

# Register the dataset
val_img_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_val/"
DatasetCatalog.register("russian_price_labels_val", lambda: get_pricetag_dicts(val_img_dir))
MetadataCatalog.get("russian_price_labels_val").set(thing_classes=["-", "name", "old_price", "price", "promotion"])


# register_coco_instances("russian_price_labels_train", {},
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/_annotations.coco.json",
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/train")
# register_coco_instances("russian_price_labels_val", {},
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_val/_annotations.coco.json",
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_val/valid")

# Configuration setup
cfg = get_cfg()
cfg.merge_from_file("./configs/faster_rcnn_R_50_C4_3x.yaml")
cfg.DATASETS.TRAIN = ("russian_price_labels_train",)
cfg.DATASETS.TEST = ("russian_price_labels_test",)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = "/home/karun95/master_thesis/solo-learn/downstream/object_detection/detectron_model.pkl"  # Path to the pre-trained BYOL model
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Verify dataset registration
assert "russian_price_labels_train" in DatasetCatalog.list(), "russian_price_labels_train not registered!"
assert "russian_price_labels_test" in DatasetCatalog.list(), "russian_price_labels_test not registered!"

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

# Create evaluator
evaluator = COCOEvaluator("russian_price_labels_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "russian_price_labels_val")

# Run evaluation
print(inference_on_dataset(trainer.model, val_loader, evaluator))
