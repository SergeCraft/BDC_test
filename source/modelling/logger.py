from clearml import Task
from train_config import Config


class ClearMLLogger:
    def __init__(self, config: Config):
        task = Task.init(
            project_name=config.project_name,
            task_name=config.experiment_name,
        )
        task.connect(config.to_dict())
        self.logger = task.get_logger()
