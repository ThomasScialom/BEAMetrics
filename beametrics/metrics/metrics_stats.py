from typing import List, Dict
from collections import Counter
import spacy

from beametrics.metrics.metrics import MetricBase

class MetricLength(MetricBase):

    def __init__(
            self,
            lang: str,
            task: str,
            *args, **kwargs
    ):
        # todo: make it mutlilingual with lang
        self.spacy_pipeline = spacy.load('en_core_web_sm')

    @classmethod
    def metric_name(cls):
        return 'length'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    )-> Dict:
        return {
            self.metric_name(): [len(self.spacy_pipeline(text)) for text in predictions]
        }

class MetricAbstractness(MetricBase):

    def __init__(
            self,
            lang: str,
            task: str,
            *args, **kwargs
    ):
        # todo: make it mutlilingual with lang
        self.spacy_pipeline = spacy.load('en_core_web_sm')
        self.Ns = (1, 2, 3)

    @classmethod
    def metric_name(cls):
        return 'abstractness'

    def sub_metric_names(self):
        return [f'{self.metric_name()}_{N}' for N in self.Ns]

    def ngrams(self, text, n):
        tokens = [t.text for t in self.spacy_pipeline(text)]
        output = {}
        for i in range(len(tokens) - n + 1):
            g = ' '.join(tokens[i:i + n])
            output.setdefault(g, 0)
            output[g] += 1
        return output

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    )-> Dict:
        res = {}
        for N in self.Ns:
            novel_grams = []
            for src, pred in zip(sources, predictions):
                pred_N_grams = self.ngrams(pred, N)
                src_N_grams = set(self.ngrams(src, N))

                ex_novel_grams = 0
                for tok in pred_N_grams:
                    if tok not in src_N_grams:
                        ex_novel_grams += 1
                novel_grams.append(
                    ex_novel_grams
                    /(len(pred_N_grams)+ 1e-5)
                )
            res[f'{self.metric_name()}_{N}'] = novel_grams

        return res

class MetricRepetition(MetricBase):
    def __init__(
            self,
            lang: str,
            task: str,
            *args, **kwargs
    ):
        # todo: make it mutlilingual with lang
        self.spacy_pipeline = spacy.load('en_core_web_sm')
        self.Ns = (1, 2, 3)

    @classmethod
    def metric_name(cls):
        return 'repetition'

    def sub_metric_names(self):
        return [f'{self.metric_name()}_{N}' for N in self.Ns]

    def ngrams(self, text, n):
        tokens = [t.text for t in self.spacy_pipeline(text)]
        output = {}
        for i in range(len(tokens) - n + 1):
            g = ' '.join(tokens[i:i + n])
            output.setdefault(g, 0)
            output[g] += 1
        return output

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    )-> Dict:

        res = {}
        for N in self.Ns:
            reps = []
            for pred in predictions:
                count_toks = Counter(self.ngrams(pred, N))
                reps.append(
                    sum(count_toks.values()) - len(count_toks)
                    / (len(count_toks) + 1e-5)
                )
            res[f'{self.metric_name()}_{N}'] = reps

        return res
