import lightning as pl
import torch
from timm import create_model

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object


class {{cookiecutter.project_shortcut}}Module(pl.LightningModule):
    def __init__(self, config: Config):
        """Inits Poster module with specified config

        Args:
            config (Config): configuration of the learning process
        """
        super().__init__()
        self._config = config

        self._model = create_model(num_classes=self._config.num_classes, **self._config.model_kwargs)
        self._losses = get_losses(self._config.losses)

        raise NotImplementedError('Implement logic of getting metrics')

        metrics = get_metrics(
            # Get your metrics
        )

        self._valid_metrics = metrics.clone(prefix='val_')
        {%- if cookiecutter.train_val_test_split %}
            self._test_metrics = metrics.clone(prefix='test_')
        {% endif %}
        
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): input Tensor

        Returns:
            torch.Tensor: predicted model values
        """
        return self._model(x)

    def configure_optimizers(self) -> dict:
        """Configures optimizers with the given config file

        Returns:
            dict: dict with optimizer and scheduler
        """
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step of the model that includes forward pass and loss calculation

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): tuple of images batch and labels batch
            batch_idx (int): index of the current batch

        Returns:
            torch.Tensor: calculated loss value
        """
        
        images, gt_labels = batch
        pr_logits = self(images)

        return self._calculate_loss(pr_logits, gt_labels, 'train_')

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step of the model that includes forward pass, loss and metrics calculation

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): tuple of images batch and labels batch
            batch_idx (int): index of the current batch
        """
        images, gt_labels = batch
        pr_logits = self(images)
        self._calculate_loss(pr_logits, gt_labels, 'val_')
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, gt_labels)

    {%- if cookiecutter.train_val_test_split %}
        def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step of the model that includes forward pass and metrics calculation

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): tuple of images batch and labels batch
            batch_idx (int): index of the current batch
        """
        images, gt_labels = batch
        pr_logits = self._model(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)
    {% endif %}

    def on_validation_epoch_start(self) -> None:
        """Reset metrics before validation epoch
        """
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log metrics after validation epoch
        """
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    {%- if cookiecutter.train_val_test_split %}
        def on_test_epoch_start(self) -> None:
            """Reset metrics before test epoch
            """
            self._test_metrics.reset()

        def on_test_epoch_end(self) -> None:
            """Log metrics after test epoch
            """
            self.log_dict(self._test_metrics.compute(), on_epoch=True)
    {% endif %}

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """Help function for calculating loss value from multiple losses

        Args:
            pr_logits (torch.Tensor): model output
            gt_labels (torch.Tensor): ground truth values
            prefix (str): prefix for loss logging

        Returns:
            torch.Tensor: calculated loss
        """
        total_loss: torch.Tensor = 0

        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, gt_labels)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        
        self.log(f'{prefix}total_loss', total_loss.item())
        
        return total_loss
