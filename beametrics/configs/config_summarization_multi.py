import pandas as pd
import os
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class SummarizationMultiSummEval(ConfigBase):
    def __init__(self):

        file_name = 'Multi_SummEval'
        file_name_processed = 'processed.multi_summeval'
        metric_names = _DEFAULT_METRIC_NAMES #+ _DEFAULT_METRIC_NAMES_SRC

        name_dataset = 'SummmEval-multi'
        short_name_dataset = 'mSu'
        languages = ['de', 'es', 'fr', 'ru', 'tr', 'en', 'zh', 'id']

        task = "summarization"
        number_examples = 2160
        nb_refs = 1
        dimensions_definitions = {
            "focus": "How much information contained in the evaluated summary text can also be found in the source document?",
            "coverage": "How much information contained in the source document can also be found in the evaluated summary?"
        }
        scale = "likert"
        source_eval_sets = "MLSUM ('de', 'es', 'fr', 'ru', 'tr'), CNN/DailyMail (en), Liputan6 (id), and LCSTS (zh)"
        annotators = "3 Amazon Mechanical Turk annotators"
        additional_comments = """Direct Assessment (“DA”) method (Graham et al., 2015; Graham et al., 2017), which has become the de facto for MT evaluation in WMT. For each HIT (100 samples), DA incorporates 10 pre-annotated samples for quality control. Crowd-sourced workers are given two texts and asked the question (in the local language): How much information contained in the second text can also be found in the first text? We combine focus and coverage annotation into 1 task, as the only thing that differentiates them is the ordering of the system and reference summaries, which is opaque to the annotators."""
        sampled_from = "https://arxiv.org/pdf/2106.01478.pdf"
        citation = """@article{koto2021evaluating,
        title={Evaluating the Efficacy of Summarization Evaluation across Languages},
        author={Koto, Fajri and Lau, Jey Han and Baldwin, Timothy},
        journal={arXiv preprint arXiv:2106.01478},
        year={2021}
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
        d_data = dict()
        for lang in ['DE', 'ES', 'FR', 'RU', 'TR', 'EN', 'ZH', 'ID']:

            lang_path = os.path.join(path, lang)

            df_score = pd.read_csv(os.path.join(lang_path, 'score.csv'))

            """
            if lang in {'DE', 'ES', 'FR', 'RU', 'TR'}:
                if lang == 'TR':
                    key_lang = 'tu'
                else:
                    key_lang = lang.lower()
                original_dataset = load_dataset('mlsum', key_lang)
            """

            for i, item in df_score.iterrows():

                item_id = item.id
                if lang == "ZH":
                    item_id = (6 - len(str(item.id))) * '0' + str(item.id)

                # reference
                with open(os.path.join(lang_path, 'gold', str(item_id)), 'r') as f:
                    gold = ' '.join(l.strip() for l in f.readlines())

                """
                # source
                assert gold[:30] == original_dataset['test'][item.id]['summary'].lower()[:30]
                source = original_dataset['test'][item.id]['text']
                """

                # prediction
                key = 'pred_BERT' if item.model == 'BERT' else 'pred_PG'
                with open(os.path.join(lang_path, key, str(item_id)), 'r') as f:
                    pred = ' '.join(l.strip() for l in f.readlines())

                d_data[f'{lang}.{i}'] = {
                    'source': None,
                    'references': [gold],
                    'hypothesis': pred,
                    'focus': item.focus,
                    'coverage': item.coverage,
                    'model': item.model,
                    'language': lang
                }

        return d_data
