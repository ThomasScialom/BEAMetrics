from beametrics.metrics.metrics_hugging_face import *
from beametrics.metrics.metrics_stats import *
#from beametrics.metrics.metrics_questeval import *

_D_METRICS = {
    MetricLength.metric_name(): MetricLength,
    MetricAbstractness.metric_name(): MetricAbstractness,
    MetricRepetition.metric_name(): MetricRepetition,
    MetricSacreBleu.metric_name(): MetricSacreBleu,
    MetricRouge.metric_name(): MetricRouge,
    MetricMeteor.metric_name(): MetricMeteor,
    MetricSari.metric_name(): MetricSari,
    MetricBertscore.metric_name(): MetricBertscore,
    MetricBleurtScore.metric_name(): MetricBleurtScore,
    #MetricQuestEval.metric_name(): MetricQuestEval,
}