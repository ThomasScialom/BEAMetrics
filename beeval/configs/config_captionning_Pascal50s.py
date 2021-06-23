from typing import List, Dict
import pickle
from beeval.configs.config_base import ConfigBase
from beeval.metrics.metric_reporter import _DEFAULT_METRIC_NAMES

class CaptioningPascal50s(ConfigBase):
    def __init__(self):

        file_name = 'captioning_human_judgments.pkl'
        file_name_processed = 'processed.captioning.pascal50s'
        metric_names = _DEFAULT_METRIC_NAMES
        language = "en"
        task = "captioning"
        nb_refs = 50

        dimensions = ('score', )

        dimensions_definitions = {'score': "triplet (A,B,C), where A is composed of 50 reference captions, "
                                           "B and C are two candidate captions. Annotators were asked to chose between B and C "
                                           "the more appropriate caption for the corresponding given image compared to A",
        }

        scale = "pairwise"

        sampled_from = "http://vrama91.github.io/cider/"

        citation = """
                        @inproceedings{vedantam2015cider,
                        title={Cider: Consensus-based image description evaluation},
                        author={Vedantam, Ramakrishna and Lawrence Zitnick, C and Parikh, Devi},
                        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
                        pages={4566--4575},
                        year={2015}}
                        """

        additional_comments = ""

        super().__init__(
            file_name=file_name,
            file_name_processed=file_name_processed,
            metric_names=metric_names,
            language=language,
            task=task,
            nb_refs=nb_refs,
            dimensions=dimensions,
            dimensions_definitions=dimensions_definitions,
            scale=scale,
            sampled_from=sampled_from,
            citation=citation,
            additional_comments=additional_comments
        )

    def format_file(
        self,
        path
    ):
        keep_only = 'pascal50s'
        # Read the data
        with open(path, 'rb') as pickle_file:
            file_log = pickle.load(pickle_file)
            system = file_log[keep_only]

        d_data = dict()
        for i in range(len(system['score'])):
            ex = {
                'source': None,
                'references': system['reference'][i],
                'hypothesis': system['candidate'][i],
                'score': system['score'][i],
            }
            d_data[i] = ex

        return d_data

    def fill_rank(
            self,
            d_data: Dict,
            dim_1s: List[str],
            dim_2s: List[str]
    ):

        dim_1s = self.dimensions
        for ex_id in range(0, len(d_data), 2):
            for metric in set(dim_1s + dim_2s):
                m1 = d_data[str(ex_id)][metric]
                m2 = d_data[str(ex_id + 1)][metric]

                if m1 >= m2:
                    rank1 = 1
                    rank2 = 0
                else:
                    rank1 = 0
                    rank2 = 1

                d_data[str(ex_id)][metric] = rank1
                d_data[str(ex_id + 1)][metric] = rank2

        return d_data