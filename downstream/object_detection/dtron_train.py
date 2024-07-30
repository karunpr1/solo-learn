from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.engine import DefaultPredictor


import os
import pickle
import json
from dtron_utils import *


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    Custom ROI heads class with an extra normalization layer after the Res5 block.
    As described in the MOCO paper, there is an extra BN layer following the res5 stage.
    """

    def _build_res5_block(self, cfg):
        """
        Build the Res5 block with an additional normalization layer.

        Args:
            cfg (CfgNode): Configuration node containing model parameters.

        Returns:
            nn.Sequential: The sequential Res5 block with an added normalization layer.
            int: Output channels of the Res5 block.
        """
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


config_file_path = "configs/faster_rcnn_R_50_C4_1x.yaml"
pretrained_weights = "./detectron_model.pkl"

output_dir = "./new_output"
num_classes = 5

device = "cuda"  # "cpu"

cfg_save_path = "dtron_OD_config.pickle"

train_dataset_name = "russian_price_labels_train"
train_data_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/"
train_images_path = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/images"
train_annotation_path = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_train/_annotations.coco.json"

test_dataset_name = "russian_price_labels_test"
test_data_dir = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/"
test_images_path = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/images"
test_annotation_path = "/home/karun95/master_thesis/datasets/russian_price_labels/price_labels_test/_annotations.coco.json"

DatasetCatalog.register(train_dataset_name, lambda: get_pricetag_dicts(train_data_dir))
MetadataCatalog.get(train_dataset_name).set(thing_classes=["-", "name", "old_price", "price", "promotion"])

DatasetCatalog.register(test_dataset_name, lambda: get_pricetag_dicts(test_data_dir))
MetadataCatalog.get(test_dataset_name).set(thing_classes=["-", "name", "old_price", "price", "promotion"])

plot_samples(dataset_name=train_dataset_name, n=5)


def main():
    cfg = get_train_cfg(config_file_path, pretrained_weights, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)

    trainer.train()

    predictor = DefaultPredictor(cfg)
    coco_evaluator(cfg, predictor, test_dataset_name)


if __name__ == "__main__":
    main()
