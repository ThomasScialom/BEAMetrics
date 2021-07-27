import logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger()
component_logger = logger.getChild("BEAMetrics")
component_logger.setLevel(logging.INFO)


import json
import pandas as pd
from beametrics.configs import *
def print_final_tables():


    D_ALL_DATASETS = {
        'SummarizationCNNDM': {
            'class': SummarizationCNNDM,
            'map_dim': {'readability': 'fluency', 'sementics': 'consistency'}
        },

        # 'SummarizationMultilingual': None,

        # 'RealSum': None,

        'SimplificationLikertSystem': {
            'class': SimplificationLikertSystem,
            'map_dim': {'readability': 'fluency', 'sementics': 'meaning'}
        },

        'Data2textWebNLG': {
            'class': Data2textWebNLG,
            'map_dim': {'readability': 'fluency', 'sementics': 'semantics'}
        },

        'CaptioningFlickr8k': {
            'class': CaptioningFlickr8k,
            'map_dim': {'readability': 'score', 'sementics': 'score'}
        },

        'CaptioningPascal50s': {
            'class': CaptioningPascal50s,
            'map_dim': {'readability': 'score', 'sementics': 'score'}
        },

        # 'NeurIPSQA': None,

        # 'OKVQA': None,
    }

    path = "data/correlation/"

    d_scores_global = {
        'readability': {'max': {}, '1': {}, "0(QE)": {}},
        'sementics': {'max': {}, '1': {}, "0(QE)": {}}
    }

    for dataset in D_ALL_DATASETS:

        if D_ALL_DATASETS[dataset] != None:
            config = D_ALL_DATASETS[dataset]['class']()
            file_name_processed = config.file_name_processed
            nb_ref_max = config.nb_refs

            with open(path + file_name_processed, 'r') as f:
                correl_file = json.load(f)

        for type_score in ['readability', 'sementics']:

            key_score = D_ALL_DATASETS[dataset]['map_dim'][type_score]

            for nb_ref, nb_ref_name in [(1, "1"), (nb_ref_max, 'max')]:
                key_nb_ref = f'ref_{str(nb_ref)}'
                d_scores_global[type_score][nb_ref_name][dataset] = {
                    'ROUGE1': correl_file[key_nb_ref]['d_scores'][key_score]['rouge1'],
                    'ROUGE2': correl_file[key_nb_ref]['d_scores'][key_score]['rouge2'],
                    'ROUGEL': correl_file[key_nb_ref]['d_scores'][key_score]['rougeL'],
                    'BLEU': correl_file[key_nb_ref]['d_scores'][key_score]['sacrebleu'],
                    'METEOR': correl_file[key_nb_ref]['d_scores'][key_score]['meteor'],
                    'SARI': 'Nan',
                    'BERTScore': correl_file[key_nb_ref]['d_scores'][key_score]['bertscore'],
                    'BLEURT': correl_file[key_nb_ref]['d_scores'][key_score]['bleurt'],
                }

            d_scores_global[type_score]["0(QE)"][dataset] = {
                'Length': 'Nan',
                'Novelty': 'Nan',
                # 'SUMQA': 'Nan',
            }

    # print the results
    for type_score in ['readability', 'sementics']:
        dfs = []
        for nb_ref_name in ['max', "1", "0(QE)"]:
            dfs.append(pd.DataFrame(d_scores_global[type_score][nb_ref_name]))

        print('_____ TYPE SCORE: ', type_score, "('nb_ref_name': ['max', '1', '0(QE)']) _____\n\n")
        print(pd.concat(dfs))
