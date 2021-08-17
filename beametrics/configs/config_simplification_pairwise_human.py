import numpy as np
import pandas as pd
import ast
import copy
from typing import Dict, Tuple
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class SimplificationPairwiseHuman(ConfigBase):
    def __init__(self):

        file_name = 'asset_pairwise_human_comparisons.csv'
        file_name_processed = 'processed.simplification.pairwise_human'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC
        metric_names = metric_names + ('sari',)

        language = "en"
        task = "simplification"
        number_examples = -1
        nb_refs = 10
        dimensions_definitions = {
            'simplicity': "to what extent is the evaluated text easier to read and understand?",
            'meaning': "how well the evaluated text expresses the original meaning?",
            'fluency': "how fluent is the evaluated text?"
            }

        scale = "pairwise"
        source_eval_sets = "3 different dataset including ASSET"
        annotators = "Mechanical Turkers"
        additional_comments = "The evaluated texts are written by humans."
        sampled_from = "https://www.aclweb.org/anthology/2020.acl-main.424/"
        citation = """@inproceedings{alva2020asset, 
                            title={ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations},
                            author={Alva-Manchego, Fernando and Martin, Louis and Bordes, Antoine and Scarton, Carolina and Sagot, Beno{\^\i}t and Specia, Lucia},
                            booktitle={ACL 2020-58th Annual Meeting of the Association for Computational Linguistics},
                            year={2020}}"""


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
        do_avg_raters = True
        keep_only = 'human'

        df = pd.read_csv(path)

        d_data = {}
        for i, row in df.iterrows():

            # drop the turkcorpus samples since they are written by humans
            if keep_only == 'system' and 'turkcorpus' in row['comparison_id']:
                continue

            unique_id = f"{row['comparison_id']}_{row['candidate_name']}"
            if unique_id not in d_data:
                d_data[unique_id] = {'source': row['source'],
                                     'hypothesis': row['simplification'],
                                     'references': ast.literal_eval(row['references']),
                                     'simplicity': [],
                                     'meaning': [],
                                     'fluency': []
                                     }

            if row['winner'] == 'similar':
                rank = 0.5
            else:
                _, m1, m2 = row['comparison_id'].split('_')
                assert m1 == row['winner'] or m2 == row['winner']
                if row['candidate_name'] == row['winner']:
                    rank = 1
                else:
                    rank = 0

            d_data[unique_id][row['aspect']].append(rank)

        if do_avg_raters:
            for k in d_data.keys():
                for dim in self.dimensions:
                    if len(d_data[k][dim]) > 0:
                        pass
                    d_data[k][dim] = np.average(d_data[k][dim])

        return d_data

    def fill_rank(
        self,
        d_data: Dict[str, Dict],
        ref_key: str,
        dim_humans: Tuple[str],
        dim_metrics: Tuple[str]
    ):

        def get_rank(v1, v2):
            if v1 > v2:
                rank = 1
            elif v2 == v2:
                rank = 0.5
            else:
                rank = 0.0
            return rank

        copy_d_data = copy.deepcopy(d_data)

        for ex_id, ex in copy_d_data.items():
            l, m1, m2, m_ex = ex_id.split('_')
            m_paired = m1 if m1 != m_ex else m2
            paired_key = f'{l}_{m1}_{m2}_{m_paired}'

            model_name = ex_id.split('.')[0].split('_')[1]
            d_data[ex_id]['model_name'] = model_name

            for dim_h in dim_humans:
                v1 = ex[dim_h]
                v2 = copy_d_data[paired_key][dim_h]
                d_data[ex_id][dim_h] = get_rank(v1, v2)

            for dim_m in dim_metrics:
                v1 = ex[ref_key][dim_m]
                v2 = copy_d_data[paired_key][ref_key][dim_m]
                d_data[ex_id][ref_key][dim_h] = get_rank(v1, v2)

        return d_data
