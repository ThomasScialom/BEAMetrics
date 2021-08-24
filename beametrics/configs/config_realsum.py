import pickle as pkl
import os

from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class REALSum_eval(ConfigBase):
    def __init__(self):

        file_name = 'realsum/'
        file_name_processed = 'processed.realsum.json'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC

        name_dataset = 'RealSum'
        short_name_dataset = 'Rea'
        languages = ["en"]
        task = "summarization"
        number_examples = 2500
        nb_refs = 1
        dimensions_definitions = {
            'litepyramid_recall': "A lightweight sampling-based version of the Pyramid that is crowdsourcable. See Crowdsourcing Lightweight Pyramids for Manual Summary Evaluation",
        }
        scale = "likert"
        source_eval_sets = "CNNDM"
        annotators = "Expert annotators working at the Sorbonne University."
        additional_comments = "100 documents for which 14 abstractive & 11 extractive summaries are annotated following the pyramid method"
        sampled_from = "https://github.com/neulab/REALSumm"
        citation = """@inproceedings{Bhandari-2020-reevaluating,
        title = "Re-evaluating Evaluation in Text Summarization",
        author = "Bhandari, Manik  and Narayan Gour, Pranav  and Ashfaq, Atabak  and  Liu, Pengfei and Neubig, Graham ",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        year = "2020"
}"""

        super().__init__(
            file_name=file_name,
            file_name_processed=file_name_processed,
            metric_names=metric_names,
            name_dataset=name_dataset,
            short_name_dataset=short_name_dataset,
            languages=languages,
            task=task,
            nb_refs=nb_refs,
            number_examples=number_examples,
            dimensions_definitions=dimensions_definitions,
            scale=scale,
            sampled_from=sampled_from,
            source_eval_sets=source_eval_sets,
            annotators=annotators,
            citation=citation,
            additional_comments=additional_comments
        )

    def format_file(
        self,
        path,
        model_detail = True
    ):

        def get_pickle(file_path):
            with open(file_path, 'rb') as fp:
                x = pkl.load(fp)
            return x

        cnndm_indexes = [1017, 10586, 11343, 1521, 2736, 3789, 5025, 5272, 5576, 6564, 7174, 7770, 8334, 9325, 9781,
                         10231, 10595, 11351, 1573, 2748, 3906, 5075, 5334, 5626, 6714, 7397, 7823, 8565, 9393, 9825,
                         10325, 10680, 11355, 1890, 307, 4043, 5099, 5357, 5635, 6731, 7535, 7910, 8613, 9502, 10368,
                         10721, 1153, 19, 3152, 4303, 5231, 5420, 5912, 6774, 7547, 8001, 8815, 9555, 10537, 10824,
                         1173, 1944, 3172, 4315, 5243, 5476, 6048, 6784, 7584, 8054, 8997, 9590, 10542, 11049, 1273,
                         2065, 3583, 4637, 5244, 5524, 6094, 6976, 7626, 8306, 9086, 9605, 10563, 11264, 1492, 2292,
                         3621, 4725, 5257, 5558, 6329, 7058, 7670, 8312, 9221, 9709]

        """
        with open(os.path.join(path, 'cnndm.ref.txt'), 'r') as f_src:
            refs = f_src.readlines()
        """

        with open(os.path.join(path, 'cnndm.src.txt'), 'r') as f_src:
            srcs = f_src.readlines()

        clean_format = lambda x: x.replace('<t> ', '').replace(' </t>', '')

        d_data = {}

        for name_file in ['abs', 'ext']:
            raw_data = get_pickle(os.path.join(path, f'REALSumm_scores_dicts_{name_file}.pkl'))

            for ex_id, ex in raw_data.items():

                hyps = ex['system_summaries']
                for name_model, d_model in hyps.items():
                    d_data[f'{name_file}_{ex_id}_{name_model}'] = {
                        'source': srcs[cnndm_indexes[ex['doc_id']]],
                        'hypothesis': clean_format(d_model['system_summary']),
                        'references': [clean_format(ex['ref_summ'])],
                        'litepyramid_recall': d_model['scores']['litepyramid_recall'],
                        'name_model': name_model
                    }

        return d_data


