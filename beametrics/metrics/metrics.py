from typing import List, Dict

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

    def sub_metric_names(self):
        return [self.metric_name()]
