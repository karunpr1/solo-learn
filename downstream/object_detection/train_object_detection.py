# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copied from https://github.com/facebookresearch/moco/blob/main/detection/train_net.py

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.data.datasets import register_coco_instances
import torch
import torch.distributed as dist


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


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            print("assert VOC error")
            # assert "voc" in dataset_name
            # return PascalVOCDetectionEvaluator(dataset_name)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    register_coco_instances("price_labels_train", {},
                            "/home/karun/detectron2/datasets/price_labels/price_labels_train/_annotations.coco.json",
                            "/home/karun/detectron2/datasets/price_labels/price_labels_train")
    register_coco_instances("price_labels_val", {},
                            "/home/karun/detectron2/datasets/price_labels/price_labels_val/_annotations.coco.json",
                            "/home/karun/detectron2/datasets/price_labels/price_labels_val")
    device = torch.device('cuda:0')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:29500", world_size=1, rank=0, )
    # dist.init_process_group(backend='nccl', init_method="env://localhost:12355", world_size=1, rank=0)
    # torch.cuda.set_per_process_memory_fraction(0.8, 0)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
