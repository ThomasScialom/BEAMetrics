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
        """
        pipeline takes as input a list of predictions, sources, list_references,
        where len(predictions) corresponds to the number of examples,
        and len(list_references[0]) corresponds to the number of references per example.

        The method has to return a dictionary:
            {
                name_metric_1: [score_metric_1_ex_1, score_metric_1_ex_n],
                name_metric_2: [score_metric_1_ex_1, score_metric_1_ex_n],
                ...
            }
        where name_metric_1 and name_metric_2 are different output of the computed metric (e.g. ROUGE-1, ROUGE-2, ROUGE-L),
        and their values are a list of corresponding scores for each example from 1 to n with n = len(predictions).
        """
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



