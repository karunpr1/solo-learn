from detectron2.engine import DefaultPredictor
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.data.datasets import register_coco_instances

import os
import pickle

from dtron_utils import *

cfg_load_path = "dtron_OD_config.pickle"

test_dataset_name = "russian_price_labels_test"
test_data_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/"
test_images_path = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/images"
test_annotation_path = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/_annotations.coco.json"

DatasetCatalog.register(test_dataset_name, lambda: get_pricetag_dicts(test_data_dir))
MetadataCatalog.get(test_dataset_name).set(thing_classes=["-", "name", "old_price", "price", "promotion"])

class_names = ["product", "name", "old_price", "price", "promotion"]
metadata = {"thing_classes": class_names}

with open(cfg_load_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


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


predictor = DefaultPredictor(cfg)

test_image(test_dataset_name, predictor, n=1, threshold=0.8)

# coco_evaluator(cfg, predictor, test_dataset_name)
