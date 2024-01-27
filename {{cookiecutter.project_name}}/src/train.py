import argparse
import logging
import os

import lightning as pl
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from clearml import Task

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import {{cookiecutter.project_shortcut}}DM
from src.lightning_module import {{cookiecutter.project_shortcut}}Module


def arg_parse() -> argparse.Namespace:
    """Parses CLI input

    Returns:
        argparse.Namespace: parse result
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')

    return parser.parse_args()


def train(config: Config):
    """Main train function that configs all training process

    Args:
        config (Config): configuration object for training
    """
    datamodule = {{cookiecutter.project_shortcut}}DM(config.data_config)
    model = {{cookiecutter.project_shortcut}}Module(config)

    # Configure ClearML
    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.model_dump())

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    # Configure Lightning components
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=3,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=config.device,
        log_every_n_steps=20,
        deterministic=True,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=4, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        logger=loggers.TensorBoardLogger(experiment_save_path, log_graph=True),
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)
    {% if cookiecutter.train_val_test_split %}
        trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)
    {% endif %}


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)

    pl.seed_everything(config.random_seed, workers=True)
    
    train(config)
