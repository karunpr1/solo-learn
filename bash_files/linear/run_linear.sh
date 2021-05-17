python3 ../../main_linear.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --epochs 100 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 10 \
    --name simclr-linear-eval \
    --dali \
    --pretrained_feature_extractor ../pretrain/trained_models/1pqoqcxo \
    --project contrastive_learning \
    --wandb