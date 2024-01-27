from typing import Optional, Union

import albumentations as albu
import cv2
import numpy as np

from torch import Tensor

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class {{cookiecutter.project_shortcut}}Dataset(Dataset):
    def __init__(
        self,
        images_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        """Inits Planet dataset

        Args:
            images_folder (str): directory where images were saved
            transforms (Optional[TRANSFORM_TYPE], optional): sequence of transforms that should be applied. Defaults to None.
        """
        self.images_folder = images_folder
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[Tensor, np.ndarray]:
        """Returns image and labels

        Args:
            idx (int): index of the object

        Returns:
            tuple[Tensor, np.array]: image as Tensor and labels in numpy array 
        """
        raise NotImplementedError('Implement logic of getting labels and images')

        image_path = # Form image path
        labels = # Your labels

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {'image': image, 'labels': labels}

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['labels']

    def __len__(self) -> int:
        """Returns length of the dataset

        Returns:
            int: length of the dataset
        """
        raise NotImplementedError('Implement logic of getting size of the dataset')
