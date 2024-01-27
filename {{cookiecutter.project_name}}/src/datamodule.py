import os
from typing import Optional

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import {{cookiecutter.project_shortcut}}Dataset
from src.logger import init_logger


logger = init_logger(__name__)


class {{cookiecutter.project_shortcut}}DM(LightningDataModule):
    def __init__(self, config: DataConfig):
        """Inits Planet data module with specified config

        Args:
            config (DataConfig): config with data loading settings
        """
        super().__init__()
        self._batch_size = config.batch_size
        self._n_workers = config.n_workers
        self._train_fraq = config.train_fraq
        self._dataset_path = config.dataset_path
        self._train_transforms = get_transforms(width=config.width, height=config.height, augment_cfg_list=config.augmentations)
        self._valid_transforms = get_transforms(width=config.width, height=config.height augmentations=False)
        {% if cookiecutter.train_val_test_split %}
            self._test_transforms = get_transforms(width=config.width, height=config.height, augmentations=False)
        {% endif %}
        self._image_folder = os.path.join(config.dataset_path, 'images')

        self.train_dataset: Dataset
        self.valid_dataset: Dataset
        {% if cookiecutter.train_val_test_split %}
            self.test_dataset: Dataset
        {% endif %}
    
    def prepare_data(self) -> None:
        """Checks data existance and log datasets info. 
        It will be called once before any worker start
        """
        raise NotImplementedError('Define your preprocessing logic')

    def setup(self, stage: Optional[str] = None):
        """Setups each worker environment for training and validation purposes

        Args:
            stage (Optional[str], optional): specifies stage of the model training (Can be 'fit' and 'test'). Defaults to None.
        """
        raise NotImplementedError('Implement dataset setup')

        if stage == 'fit':
            train_df = read_df(self._dataset_path, 'train')

            self.train_dataset = {{cookiecutter.project_shortcut}}Dataset(
                # Add your args
                images_folder=self._image_folder,
                transforms=self._train_transforms,
            )

            valid_df = read_df(self._dataset_path, 'valid')

            self.valid_dataset = {{cookiecutter.project_shortcut}}Dataset(
                # Add your args
                images_folder=self._image_folder,
                transforms=self._valid_transforms,
            )
        {% if cookiecutter.train_val_test_split %}
            elif stage == 'test':
                test_df = read_df(self._dataset_path, 'test')

                self.test_dataset = {{cookiecutter.project_shortcut}}Dataset(
                    # Add your args
                    images_folder=self._image_folder,
                    transforms=self._test_transforms,
                )
        {% endif %}

    def train_dataloader(self) -> DataLoader:
        """Returns dataloader with training data

        Returns:
            DataLoader: loader with training data
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns dataloader with validation data

        Returns:
            DataLoader: loader with validation data
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
    
    {% if cookiecutter.train_val_test_split %}
        def test_dataloader(self) -> DataLoader:
            """Returns dataloader with validation data

            Returns:
                DataLoader: loader with validation data
            """
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self._batch_size,
                num_workers=self._n_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )
    {% endif %}


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """Reads dataframe after splitting from the specified data path

    Args:
        data_path (str): path to the csv file
        mode (str): string value for getting train, val or test datasets

    Returns:
        pd.DataFrame: dataframe loaded from disk
    """
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
