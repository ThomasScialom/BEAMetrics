# todo: from easse.fkgl import corpus_fkgl
from questeval.questeval_metric import QuestEval

from typing import List, Dict
import numpy as np
from datasets import load_metric

class MetricBase():
    """
    To add a new metric it should have a method pipeline returning a dict
    with the scores name as keys and the list of corresponding scores as values.
    """
    def __init__(self):
        raise NotImplementedError

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict[str, List[float]]:
        raise NotImplementedError

    @classmethod
    def metric_name(cls):
        return cls.__name__

class MetricBaseHF(MetricBase):
    def __init__(self, *args, **kwargs):
        """
        Metric Class that have a pipeline implemented
        for any metrics from the Hugging Face library.
        """
        raise NotImplementedError

    def pipeline(
        self,
        predictions: List[str] = None,
        sources: List[str] = None,
        list_references: List[List[str]] = None,
    ) -> Dict:

        if self.emulate_multirefs:
            references_per_prediction = len(list_references[0])
            if any(len(refs) != references_per_prediction for refs in list_references):
                raise ValueError("It is required to set the same number of references for each prediction")

            list_final_scores = []
            for idx in range(references_per_prediction):
                temp_references = [refs[idx] for refs in list_references]
                _final_scores = self.compute_segment_level(predictions=predictions,
                                                          sources=sources,
                                                          references=temp_references
                )
                list_final_scores.append(_final_scores)
            keys = list_final_scores[0].keys()
            final_scores = {k: [np.average([scores[k][idx] for scores in list_final_scores])
                                   for idx in range(len(predictions))] for k in keys}
        else:
            final_scores = self.compute_segment_level(predictions=predictions,
                                                      sources=sources,
                                                      references=list_references
            )

        return final_scores

    def compute_segment_level(
        self,
        predictions: List[str] = None,
        sources: List[str] = None,
        references: List = None,
    ) -> Dict:
        """

        :param predictions: List[prediction]
        :param sources: List[source] (if None, no sources available)
        :param references: can be either a List[reference] or a List[List[reference]] wrt if the metric handle mutlireferences
        :return: Dict wrt Dataset format
        """
        if sources is not None:
            assert len(predictions) == len(sources)
        if references is not None:
            assert len(predictions) == len(references)

        final_scores = []
        if self.do_seg_level:
            for idx, _ in enumerate(predictions):

                sub_predictions = [predictions[idx]]
                sub_sources = [sources[idx]] if sources is not None else None
                sub_references = [references[idx]] if references is not None else None
                final_score = self.compute(predictions=sub_predictions,
                                           sources=sub_sources,
                                           references=sub_references)
                final_scores.append(final_score)
        else:
            final_scores = self.compute(predictions=predictions,
                                       sources=sources,
                                       references=references)

        return self.get_format(final_scores)

    def compute(
        self,
        predictions: List[str] = None,
        sources: List[str] = None,
        references: List = None,
    ):
        assert isinstance(NotImplementedError, object)
        raise NotImplementedError

    def get_format(
        self,
        final_scores: Dict
    ) -> Dict[str, list]:
        raise NotImplementedError

class MetricBaseHFSrcRef(MetricBaseHF):
    def compute(
        self,
        predictions: List[str] = None,
        sources: List[str] = None,
        references: List = None,
    ):

        return self.metric.compute(predictions=predictions,
                                    references=references,
                                    sources=sources,
                                    **self.dict_kwargs
        )

class MetricBaseHFRef(MetricBaseHF):
    def compute(
        self,
        predictions: List[str] = None,
        sources: List[str] = None,
        references: List = None,
    ):

        return self.metric.compute(
            predictions=predictions,
            references=references,
            **self.dict_kwargs
        )

class MetricSari(MetricBaseHFSrcRef):

    def __init__(self, *args, **kwargs):
        self.metric = load_metric(self.metric_name())
        self.do_seg_level = True
        self.dict_kwargs = dict()
        self.emulate_multirefs = False

    @classmethod
    def metric_name(cls):
        return 'sari'

    def get_format(
        self,
        final_scores: Dict
    ) -> Dict[str, list]:
        return {'sari': [ex['sari']
                           for ex in final_scores]
                }

class MetricMeteor(MetricBaseHFRef):

    def __init__(self, *args, **kwargs):
        self.metric = load_metric(self.metric_name())
        self.do_seg_level = True
        self.dict_kwargs = dict()
        self.emulate_multirefs = True

    @classmethod
    def metric_name(cls):
        return 'meteor'

    def get_format(
            self,
            final_scores: Dict
    ) -> Dict[str, list]:
        return {'meteor': [ex['meteor']
                           for ex in final_scores]
        }

class MetricRouge(MetricBaseHFRef):

    def __init__(self, *args, **kwargs):
        self.metric = load_metric(self.metric_name())
        self.do_seg_level = False
        self.dict_kwargs = {'use_agregator': False}
        self.emulate_multirefs = True

    @classmethod
    def metric_name(cls):
        return 'rouge'

    def get_format(
            self,
            final_scores: Dict
    ) -> Dict[str, list]:
        return {name: [score[-1] for score in final_scores[name]]
                for name in final_scores
        }

class MetricSacreBleu(MetricBaseHFRef):

    def __init__(self, *args, **kwargs):
        self.metric = load_metric(self.metric_name())
        self.do_seg_level = True
        self.dict_kwargs = dict()
        self.emulate_multirefs = False

    @classmethod
    def metric_name(cls):
        return 'sacrebleu'

    def get_format(
            self,
            final_scores: Dict
    ) -> Dict[str, list]:
        return {'sacrebleu': [score['score'] for score in final_scores]}

class MetricBertscore(MetricBaseHFRef):

    def __init__(
        self,
        device: str = 'cuda',
        *args,
        **kwargs
    ):
        self.metric = load_metric(self.metric_name())
        self.do_seg_level = False
        self.dict_kwargs = {'model_type': 'bert-base-multilingual-cased', 'device': device}
        self.emulate_multirefs = False

    @classmethod
    def metric_name(cls):
        return 'bertscore'

    def get_format(
            self,
            final_scores: Dict
    ) -> Dict[str, list]:
        return {'bertscore': final_scores['f1']}

class MetricBleurtScore(MetricBaseHFRef):

    def __init__(self, *args, **kwargs):
        self.metric = load_metric(self.metric_name())
        self.do_seg_level = False
        self.dict_kwargs = {}
        self.emulate_multirefs = True

    @classmethod
    def metric_name(cls):
        return 'bleurt'

    def get_format(
            self,
            final_scores: Dict
    ) -> Dict[str, list]:
        return {'bleurt': final_scores['scores']}

class MetricQuestEval(MetricBase):

    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = QuestEval(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'questeval'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                sources = sources,
                list_references = list_references
        )

        return {self.metric_name(): res['ex_level_scores']}

class MetricQuestEval_t2t(MetricQuestEval):

    def __init__(
        self,
        lang: str,
        task: str
    ):
        task = 'text2text' if task is not 'data2text' else task
        self.metric = QuestEval(task=task, language=lang)

    @classmethod
    def metric_name(cls):
        return 'questeval_t2t'

class MetricQuestEval_t2t_src(MetricQuestEval_t2t):

    @classmethod
    def metric_name(cls):
        return 'questeval_t2t_src'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                sources = sources,
                list_references = None
        )

        return {self.metric_name(): res['ex_level_scores']}
