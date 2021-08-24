import pickle
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class TranslationWMT2019(ConfigBase):
    def __init__(
            self,
            lang1: str,
            lang2:str,
            number_examples: int
    ):
        file_name = f'WMT/{lang1}-{lang2}/data.pkl'
        file_name_processed = f'processed.wmt2019.{lang1}-{lang2}'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC

        name_dataset = f'WMT2019.{lang1}-{lang2}'
        short_name_dataset = f'WMT.{lang1}-{lang2}'
        languages = [lang1, lang2]
        task = "translation"
        number_examples = number_examples
        nb_refs = 1
        dimensions_definitions = {
            'DaRR': "DaRR stands for Direct Assessment from 1 to 100, followed by Relative Ranking.  'When we have at least two DA scores for translations of the same source input, it is possible to convert those DA scores into a relative ranking judgement, if the difference in DA scores allows conclusion that one translation is better than the other."
        }
        scale = "pairwise"
        source_eval_sets = "WMT2019"
        annotators = "Paid consultants, sourced by a linguistic service provider company."
        sampled_from = ""
        citation = """@inproceedings{ma2019results,
        title={Results of the WMT19 metrics shared task: Segment-level and strong MT systems pose big challenges},
        author={Ma, Qingsong and Wei, Johnny and Bojar, Ond{\v{r}}ej and Graham, Yvette},
        booktitle={Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1)},
        pages={62--90},
        year={2019}
}"""
        additional_comments = """"""

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
        path
    ):
        def read_pickle(file):
            with open(file, 'rb') as f:
                data = pickle.load(f)
            return data


        pickle_data = read_pickle(path)

        d_data = dict()
        for ex_id in pickle_data:
            ex = {
                'source': pickle_data[ex_id]['src'],
                'references': [pickle_data[ex_id]['ref']],
                'hypothesis': pickle_data[ex_id]['better']['sys'],
                'system_name': pickle_data[ex_id]['better']['sys_name'],
                'DaRR': 1,
            }
            d_data[f'{ex_id}.better'] = ex

            ex = {
                'source': pickle_data[ex_id]['src'],
                'references': [pickle_data[ex_id]['ref']],
                'hypothesis': pickle_data[ex_id]['worse']['sys'],
                'system_name': pickle_data[ex_id]['worse']['sys_name'],
                'DaRR': 0,
            }
            d_data[f'{ex_id}.worse'] = ex

        return d_data


class TranslationWMT2019_de_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='de', lang2='en', number_examples=170730)

class TranslationWMT2019_fi_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='fi', lang2='en', number_examples=64358)

class TranslationWMT2019_gu_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='gu', lang2='en', number_examples=40220)

class TranslationWMT2019_kk_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='kk', lang2='en', number_examples=19456)

class TranslationWMT2019_lt_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='lt', lang2='en', number_examples=43724)

class TranslationWMT2019_ru_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='ru', lang2='en', number_examples=79704)

class TranslationWMT2019_zh_en(TranslationWMT2019):
    def __init__(self):
        super().__init__(lang1='zh', lang2='en', number_examples=62140)

