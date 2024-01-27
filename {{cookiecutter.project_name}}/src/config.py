from typing import List, Union

from datetime import datetime
from omegaconf import OmegaConf
from pydantic import BaseModel


class AugmentationConfig(BaseModel):
    """Pydantic model class for storing augmenations parameters"""
    augm_fn: str
    augm_kwargs: dict


class DataConfig(BaseModel):
    """Pydantic model class for storing data loading parameters"""
    dataset_path: str
    batch_size: int
    n_workers: int
    train_fraq: float
    width: int
    height: int
    augmentations: List[AugmentationConfig]


class LossConfig(BaseModel):
    """Pydantic model class for storing loss functions parameters"""
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class Config(BaseModel):
    """Class for storing training parameters"""
    project_name: str
    experiment_name: str
    random_seed: int

    accelerator: str
    device: Union[int, list]

    n_epochs: int
    num_classes: int

    model_kwargs: dict

    optimizer: str
    optimizer_kwargs: dict

    scheduler: str
    scheduler_kwargs: dict
    
    losses: List[LossConfig]
    data_config: DataConfig

    monitor_metric: str
    monitor_mode: str


    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Loads config from yaml file

        Args:
            path (str): path to yaml config file

        Returns:
            Config: config object that stores all training settings
        """
        cfg: dict = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

        if 'experiment_name' not in cfg:            
            cfg['experiment_name'] = f'exp_{datetime.strftime(datetime.now(), r"%y_%m_%d__%H_%M")}'

        return cls(**cfg)
