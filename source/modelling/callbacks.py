import typing as tp
from collections import defaultdict
from time import time

import pandas as pd
from catalyst.core.callback import Callback, CallbackNode
from catalyst.core.runner import IRunner
from clearml import Logger
from train_config import Config


class ClearMLCallback(Callback):
    def __init__(self, logger: Logger, config: Config, class_names: tp.List[str], infer: bool = False):
        super().__init__(order=90, node=CallbackNode.master)  # noqa: WPS432
        self.logger = logger
        self.start_time = 0
        self.class_names = class_names
        self.infer = infer
        self.log_metrics = config.log_metrics
        self.log_metrics.extend(['loss', 'lr'])

    def on_loader_start(self, runner: IRunner) -> tp.NoReturn:
        self.start_time = time()

    def on_loader_end(self, runner: IRunner) -> tp.NoReturn:
        dt = time() - self.start_time

        self._report_scalar('time', runner.loader_key, dt, runner.global_epoch_step)

    def on_epoch_end(self, runner: IRunner) -> tp.NoReturn:
        if self.infer:
            self._lof_infer_metrics(runner)
        else:
            self._log_train_metrics(runner)
        self.logger.flush()

    def _report_scalar(self, title: str, mode: str, value: float, epoch: int) -> tp.NoReturn:
        self.logger.report_scalar(
            title=title,
            series=mode,
            value=value,
            iteration=epoch,
        )

    def _log_train_metrics(self, runner: IRunner) -> tp.NoReturn:  # noqa: WPS210
        for mode, metrics in runner.epoch_metrics.items():
            log_keys = [k for log_m in self.log_metrics for k in metrics.keys() if log_m in k]
            for k in log_keys:
                title = k
                if 'class' in k:
                    title, cl = k.split('/')
                    title = f"{title}_{self.class_names[int(cl.split('_')[1])]}"
                self._report_scalar(title, mode, metrics[k], runner.global_epoch_step)

    def _lof_infer_metrics(self, runner: IRunner) -> tp.NoReturn:  # noqa: WPS210
        infer_metircs = defaultdict(dict)
        self.log_metrics.append('support')
        metrics = runner.epoch_metrics['valid']

        log_keys = [k for log_m in self.log_metrics for k in metrics.keys() if log_m in k]
        for k in log_keys:
            if '/' in k:
                title, cl = k.split('/')
                if 'class' in cl:
                    cl = self.class_names[int(cl.split('_')[1])]
                infer_metircs[title].update({cl: metrics[k]})
        test_results = pd.DataFrame(infer_metircs)
        test_results = test_results.rename(columns={'support': 'num'}).T
        self.logger.report_table(title='Test Results', series='Test Results', iteration=0, table_plot=test_results)
