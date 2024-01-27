from dataclasses import dataclass
from typing import List

from torch import nn

from src.config import LossConfig
from src.utils import load_object


@dataclass
class Loss:
    """Loss dataclass that stores loss parameters"""
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: List[LossConfig]) -> List[Loss]:
    """Returns loss objects from list of loss configs. This function loads loss objects
    and initialises them as it defined in LossConfig object

    Args:
        losses_cfg (List[LossConfig]): list of LossConfig objects

    Returns:
        List[Loss]: list of Loss objects
    """
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        )
        for loss_cfg in losses_cfg
    ]
