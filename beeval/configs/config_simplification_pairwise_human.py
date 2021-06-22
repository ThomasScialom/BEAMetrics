import numpy as np
import pandas as pd
import ast
import copy
from typing import List, Dict, Tuple

from beeval.configs.config_base import ConfigBase

class SimplificationPairwiseHuman(ConfigBase):
    def __init__(self):

        file_name = 'asset_pairwise_human_comparisons.csv'
        file_name_processed = 'processed.simplification.pairwise_human'
        metric_names = None

        language = "en"
        task = "simplification"
        nb_refs = 10

        dimensions = ('simplicity', 'meaning', 'fluency')

        dimensions_definitions = {
            'simplicity': "to what extent is the evaluated text easier to read and understand?",
            'meaning': "how well the evaluated text expresses the original meaning?",
            'fluency': "how fluent is the evaluated text?"
            }

        scale = "pairwise"

        sampled_from = "https://www.aclweb.org/anthology/2020.acl-main.424/"

        citation = """@inproceedings{alva2020asset, 
                            title={ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations},
                            author={Alva-Manchego, Fernando and Martin, Louis and Bordes, Antoine and Scarton, Carolina and Sagot, Beno{\^\i}t and Specia, Lucia},
                            booktitle={ACL 2020-58th Annual Meeting of the Association for Computational Linguistics},
                            year={2020}}"""

        additional_comments = "The evaluated texts are written by humans."

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
        d_data: Dict,
        dim_1s: Tuple[str],
        dim_2s: Tuple[str]
    ):

        for ex_id, ex in d_data.items():
            l, m1, m2, m_ex = ex_id.split('_')
            m_paired = m1 if m1 != m_ex else m2
            paired_key = f'{l}_{m1}_{m2}_{m_paired}'
            copy_d_data = copy.deepcopy(d_data)

            for metric in set(dim_1s + dim_2s):
                if ex[metric] > copy_d_data[paired_key][metric]:
                    rank = 1
                elif ex[metric] == copy_d_data[paired_key][metric]:
                    rank = 0.5
                else:
                    rank = 0.0

                d_data[ex_id][metric] = rank

                model_name = ex_id.split('.')[0].split('_')[1]
                d_data[ex_id]['model_name'] = model_name

        return d_data
