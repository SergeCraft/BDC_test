import logging
import typing as tp

import torch
from catalyst import dl
from catalyst.core.callback import Callback
from callbacks import ClearMLCallback
from train_config import config
from const import IMAGES, LOGITS, LOSS, PREDICTS, SCORES, TARGETS, VALID
from dataset import get_class_names, get_loaders
from logger import ClearMLLogger
from models import get_model
from train_utils import set_global_seed


def get_base_callbacks(class_names: tp.List[str], infer: bool = False) -> tp.List[Callback]:
    return [
        dl.BatchTransformCallback(
            transform=torch.sigmoid,
            scope='on_batch_end',
            input_key=LOGITS,
            output_key=SCORES,
        ),
        dl.BatchTransformCallback(
            transform=lambda x: x > config.binary_thresh,
            scope='on_batch_end',
            input_key=SCORES,
            output_key=PREDICTS,
        ),
        dl.AUCCallback(input_key=SCORES, target_key=TARGETS),
        dl.MultilabelPrecisionRecallF1SupportCallback(
            input_key=PREDICTS,
            target_key=TARGETS,
            num_classes=len(class_names),
            log_on_batch=False,
        ),
        ClearMLCallback(ClearMLLogger(config).logger, config, class_names, infer=infer),
    ]


def get_train_callbacks(class_names: tp.List[str], infer: bool = False) -> tp.List[Callback]:
    callbacks = get_base_callbacks(class_names, infer)
    callbacks.extend([
        dl.CriterionCallback(input_key=LOGITS, target_key=TARGETS, metric_key=LOSS),
        dl.OptimizerCallback(metric_key=LOSS),
        dl.SchedulerCallback(loader_key=VALID, metric_key=config.valid_metric),
        dl.CheckpointCallback(
            logdir=config.checkpoints_dir,
            loader_key=VALID,
            metric_key=config.valid_metric,
            minimize=config.minimize_metric,
        ),
        dl.EarlyStoppingCallback(
            patience=config.early_stop_patience,
            loader_key=VALID,
            metric_key=config.valid_metric,
            minimize=config.minimize_metric,
        ),
    ])
    return callbacks


def train():  # noqa: WPS210
    loaders, infer_loader = get_loaders(config)
    class_names = get_class_names(config)

    model = get_model(n_classes=len(class_names), **config.model_kwargs)
    # freeze backbone except last fully-connected layer
    for module in list(model.children())[:-1]:
        module.requires_grad = False

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_kwargs)
    scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)

    runner = dl.SupervisedRunner(
        input_key=IMAGES,
        output_key=LOGITS,
        target_key=TARGETS,
    )

    runner.train(
        model=model,
        criterion=config.loss,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=get_train_callbacks(class_names),
        num_epochs=config.n_epochs,
        valid_loader=VALID,
        valid_metric=config.valid_metric,
        minimize_valid_metric=config.minimize_metric,
        seed=config.seed,
        verbose=True,
        load_best_on_end=True,
    )

    runner.evaluate_loader(
        model=model,
        loader=infer_loader['infer'],
        callbacks=get_base_callbacks(class_names, infer=True),
        verbose=True,
        seed=config.seed,
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_global_seed(config.seed)
    train()
