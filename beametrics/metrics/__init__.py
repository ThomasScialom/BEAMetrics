from beametrics.metrics.metrics import *
#from beametrics.metrics.metrics_questeval import *

_D_METRICS = {
    MetricSacreBleu.metric_name(): MetricSacreBleu,
    MetricRouge.metric_name(): MetricRouge,
    MetricMeteor.metric_name(): MetricMeteor,
    MetricSari.metric_name(): MetricSari,
    MetricBertscore.metric_name(): MetricBertscore,
    MetricBleurtScore.metric_name(): MetricBleurtScore,
    #MetricQuestEval.metric_name(): MetricQuestEval,
}