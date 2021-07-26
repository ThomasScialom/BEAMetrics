from beametrics.metrics.metrics import *
from beametrics.metrics.metrics_questeval import *

_D_METRICS = {
    MetricSacreBleu.metric_name(): MetricSacreBleu,
    MetricRouge.metric_name(): MetricRouge,
    MetricMeteor.metric_name(): MetricMeteor,
    MetricSari.metric_name(): MetricSari,
    MetricBertscore.metric_name(): MetricBertscore,
    MetricBleurtScore.metric_name(): MetricBleurtScore,
    MetricQuestEval.metric_name(): MetricQuestEval,
    MetricQuestEval_t2t.metric_name(): MetricQuestEval_t2t,
    MetricQuestEval_t2t_src.metric_name(): MetricQuestEval_t2t_src,

    MetricQuestEval_t2t_f1.metric_name(): MetricQuestEval_t2t_f1,
    MetricQuestEval_t2t_answ.metric_name(): MetricQuestEval_t2t_answ,
    MetricQuestEval_t2t_berscore.metric_name(): MetricQuestEval_t2t_berscore,
    MetricQuestEval_t2t_f1_answ.metric_name(): MetricQuestEval_t2t_f1_answ,
    MetricQuestEval_t2t_src_f1_answ.metric_name(): MetricQuestEval_t2t_src_f1_answ,

    MetricQuestEval_src_weighter.metric_name(): MetricQuestEval_src_weighter,
}