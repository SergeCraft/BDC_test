from datetime import datetime
from functools import partial

import albumentations as albu
import torch
from base_config import Config
from train_utils import preprocess_imagenet
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


SEED = 7
IMG_SIZE = 224
BATCH_SIZE = 60
N_EPOCHS = 10

augs = albu.Compose(
    [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(),
        albu.GaussNoise()
    ]
)

config = Config(
    num_workers=1,
    seed=SEED,
    loss=BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam,
    optimizer_kwargs={
        'lr': 1e-3,
        'weight_decay': 5e-4,
    },
    scheduler=ReduceLROnPlateau,
    scheduler_kwargs={
        'mode': 'min',
        'factor': 0.1,
        'patience': 5,
    },
    img_size=IMG_SIZE,
    augmentations=augs,
    preprocessing=partial(preprocess_imagenet, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    early_stop_patience=10,
    model_kwargs={'name': 'resnet18', 'pretrained': True},
    log_metrics=['auc', 'f1'],
    binary_thresh=0.1,
    valid_metric='auc',
    minimize_metric=False,
    images_dir='../../data/dataset',
    train_dataset_path='../../data/dataset/train.csv',
    valid_dataset_path='../../data/dataset/valid.csv',
    test_dataset_path='../../data/dataset/test.csv',
    valid_ratio=0.1,
    project_name='Classification-parts',
    experiment_name=f'experiment_1_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}',
)
