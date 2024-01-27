from typing import Union, List

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from src.config import AugmentationConfig
from src.utils import load_object

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    augment_cfg_list: List[AugmentationConfig] = [],
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    """Returns transforms for the dataset

    Args:
        width (int): result image width
        height (int): result image height
        preprocessing (bool, optional): defines if we should resize image. Defaults to True.
        augmentations (bool, optional): defines if we should apply augmentations. Defaults to True.
        postprocessing (bool, optional): defines if we should normalize and convert to Torch tensor. Defaults to True.

    Returns:
        TRANSFORM_TYPE: sequence of transforms
    """
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        transforms.extend(get_augmentations(augment_cfg_list))

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)


def get_augmentations(augment_cfg_list: List[AugmentationConfig]) -> List[albu.BasicTransform]:
    """Returns augmenation objects from list of augmentation configs. This function loads
    augmenations objects and initialises them as it defined in AugmentationConfig object

    Args:
        augment_cfg_list (List[AugmentationConfig]): list of AugmentationConfig objects

    Returns:
        List[albu.BasicTransform]: list of augmentation objects
    """
    return [
        load_object(augm_cfg.augm_fn)(**augm_cfg.augm_kwargs)
        for augm_cfg in augment_cfg_list
    ]

