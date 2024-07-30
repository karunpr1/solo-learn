from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
import cv2
import os
import json
import random
from detectron2.layers import get_norm


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
img_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_val/"  # Assuming the images are in the 'images' folder inside this directory
DatasetCatalog.register("russian_price_labels_val", lambda: get_pricetag_dicts(img_dir))
MetadataCatalog.get("russian_price_labels_val").set(thing_classes=["-", "name", "old_price", "price", "promotion"])


# register_coco_instances("russian_price_labels_train", {},
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/_annotations.coco.json",
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/train")
# register_coco_instances("russian_price_labels_val", {},
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_val/_annotations.coco.json",
#                         "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_val/valid")

# Load the trained model configuration
cfg = get_cfg()
cfg.merge_from_file("configs/faster_rcnn_R_50_C4_3x.yaml")
cfg.MODEL.WEIGHTS = "/home/karun95/master_thesis/solo-learn/downstream/object_detection/output/model_final.pth"  # Path to the trained model weights
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
cfg.DATASETS.TEST = ("russian_price_labels_val", )

# Create a predictor
predictor = DefaultPredictor(cfg)

# # Load test dataset
dataset_dicts = get_pricetag_dicts(img_dir)

# Run inference and visualize results on a few test images
# for d in random.sample(dataset_dicts, 10):  # Visualize a few samples randomly
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("russian_price_labels_val"), scale=0.9)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('Predicted Image', v.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


# Create evaluator
evaluator = COCOEvaluator("russian_price_labels_val", cfg, False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "russian_price_labels_val")

# Run evaluation
print(inference_on_dataset(predictor.model, test_loader, evaluator))
