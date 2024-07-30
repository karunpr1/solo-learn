from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode

import random
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import json


def get_pricetag_dicts(img_dir):
    """
    Load and parse the COCO annotations JSON file for the given image directory.

    Args:
        img_dir (str): Directory containing the image data and COCO annotations JSON file.

    Returns:
        list: A list of dictionaries, each representing an image and its annotations.
    """
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


def plot_samples(dataset_name, n=1):
    """
    Plot a random sample of images from the dataset with annotations.

    Args:
        dataset_name (str): The name of the registered dataset.
        n (int): Number of random samples to plot.
    """
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s['file_name'])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(10, 10))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, pretrained_weights, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    """
    Get the configuration for training the Detectron2 model.

    Args:
        config_file_path (str): Path to the configuration file.
        pretrained_weights (str): URL or path to the pretrained weights.
        train_dataset_name (str): Name of the training dataset.
        test_dataset_name (str): Name of the testing/validation dataset.
        num_classes (int): Number of classes in the dataset.
        device (str): Device to use for training ('cuda' or 'cpu').
        output_dir (str): Directory to save the output model and logs.

    Returns:
        CfgNode: Configuration node with the specified settings.
    """
    cfg = get_cfg()

    cfg.merge_from_file(config_file_path)
    cfg.MODEL.WEIGHTS = pretrained_weights
    cfg.DATASETS.TRAIN = (train_dataset_name, )
    cfg.DATASETS.TEST = (test_dataset_name, )

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 28000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.SOLVER_STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = output_dir

    return cfg


def test_image(dataset_name, predictor, n=2, threshold=0.5):
    """
    Perform inference on a random sample of images from the dataset and visualize the results.

    Args:
        dataset_name (str): The name of the registered dataset.
        predictor (DefaultPredictor): The Detectron2 predictor object for inference.
        n (int): Number of random samples to test.
        threshold (float): Confidence threshold for displaying predictions.
    """
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    for s in random.sample(dataset_custom, n):
        im = cv2.imread(s['file_name'])
        outputs = predictor(im)
        instances = outputs["instances"]
        scores = instances.scores
        keep = scores >= threshold
        instances = instances[keep]
        v = Visualizer(im[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(instances.to("cpu"))

        plt.figure(figsize=(10, 10))
        plt.imshow(v.get_image())
        plt.show()


def coco_evaluator(cfg, predictor, test_dataset_name):
    """
    Evaluate the model using the COCO evaluation metrics on the given test dataset.

    Args:
        cfg (CfgNode): The configuration node containing model and dataset parameters.
        predictor (DefaultPredictor): The Detectron2 predictor object for inference.
        test_dataset_name (str): The name of the registered test dataset.

    Returns:
        dict: The COCO evaluation results.
    """
    evaluator = COCOEvaluator(test_dataset_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

