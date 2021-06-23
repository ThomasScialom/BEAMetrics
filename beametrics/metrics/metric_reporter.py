from typing import List, Dict, Tuple, Any
from beametrics.utils import component_logger

from beametrics.metrics.metrics import (
    MetricBaseHFSrcRef,
    MetricBaseHFRef,
    MetricSacreBleu,
    MetricSari,
    MetricMeteor,
    MetricRouge,
    MetricBertscore,
    MetricBleurtScore,
    MetricQuestEval,
)

_DEFAULT_METRIC_NAMES = ('rouge', 'sacrebleu', 'meteor', 'bertscore', 'bleurt', 'questeval')

class MetricReporter():
    """
    Usage:
    predictions = ["Hey there", "Hey not there."]
    list_references = [["Hey hey there"], ["Hey hey not there."]]
    sources = ["Hey there", "Hey not there."]

    metricReporter = MetricReporter()
    scores = metricReporter.compute(predictions, sources, list_references)
    """

    def __init__(
        self,
        metric_names: Tuple[str] = None,
        task: str = None,
        lang: str = None,
        device: str = 'cuda',
    ):
        self.metric_names = metric_names
        if metric_names is None:
            self.metric_names = _DEFAULT_METRIC_NAMES
        self.d_metrics = {metric_name: None for metric_name in self.metric_names}

        self.task = task
        self.lang = lang
        self.device = device

    def load_metric(
        self,
        metric_name: str
    ):
        assert self.d_metrics[metric_name] is None

        component_logger.info(f"Loading {metric_name}")
        if metric_name == 'sari':
            self.d_metrics[metric_name] = MetricSari()
        if metric_name == 'sacrebleu':
            self.d_metrics[metric_name] = MetricSacreBleu()
        if metric_name == 'meteor':
            self.d_metrics[metric_name] = MetricMeteor()
        if metric_name == 'rouge':
            self.d_metrics[metric_name] = MetricRouge()
        if metric_name == 'bertscore':
            self.d_metrics[metric_name] = MetricBertscore()
        if metric_name == 'bleurt':
            self.d_metrics[metric_name] = MetricBleurtScore()
        if metric_name == 'questeval':
            self.d_metrics[metric_name] = MetricQuestEval(task=self.task, lang=self.lang)

    def get_sub_metric_names(
        self,
        metric_names: List[str] = None
    ) -> List[str]:
        if metric_names is None:
            metric_names = self.metric_names

        sub_metric_names = []
        for metric_name in metric_names:
            if metric_name == 'rouge':
                sub_metric_names += ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            else:
                sub_metric_names.append(metric_name)

        return sub_metric_names

    def get_missing_metrics(
        self,
        a_dict: Dict[str, Any],
        metric_names: List[str],
    ) -> List[str]:

        to_do_metrics = [
            metric_name for metric_name in metric_names
            if any([m not in a_dict for m in self.get_sub_metric_names([metric_name])]
            )
        ]

        return to_do_metrics

    def compute(
        self,
        predictions: List[str],
        sources: List[str] = None,
        list_references: List[List[str]] = None,
        metric_names: List[str] = None
    ) -> Dict[str, List[float]]:
        d_res = dict()

        do_src = sources is not None
        do_ref = list_references is not None
        do_src_ref = do_ref and do_src

        if metric_names is None:
            metric_names = self.metric_names
        for metric_name in metric_names:
            component_logger.info(f'Computing {metric_name}')
            if self.d_metrics[metric_name] is None:
                self.load_metric(metric_name)
            metric = self.d_metrics[metric_name]

            if isinstance(metric, MetricBaseHFSrcRef) and not do_src_ref:
                component_logger.warning(f"Skipping {metric_name}: it requires both sources and list_references.")
                continue
            if isinstance(metric, MetricBaseHFRef) and not do_ref:
                component_logger.warning(f"Skipping {metric_name}: it requires list_references.")
                continue
            scores = metric.pipeline(
                predictions=predictions,
                sources=sources,
                list_references=list_references
            )
            d_res = {**d_res, **scores}

        return d_res

















