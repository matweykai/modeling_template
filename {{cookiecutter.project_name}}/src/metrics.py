from torchmetrics import MetricCollection


def get_metrics(**kwargs) -> MetricCollection:
    """Defines metrics for measuring model performance

    Returns:
        MetricCollection: collection of metrics that we would check
    """

    raise NotImplementedError('Add your metrics')

    return MetricCollection(
        {
            # EXAMPLE:
            # from torchmetrics import F1Score, Precision, Recall
            #
            # 'f1': F1Score(**kwargs),
            # 'precision': Precision(**kwargs),
            # 'recall': Recall(**kwargs),
        }
    )