import json
import numpy as np

from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class NoVQA_eval_no_human(ConfigBase):
    def __init__(self):

        file_name = 'NoVQA_evaluation_no_human.json'
        file_name_processed = 'processed.NoVQA_evaluation_no_human'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC

        name_dataset = 'NoVQA-Eval'
        short_name_dataset = 'VQA'
        languages = ["en"]
        task = "VQA"
        number_examples = 300
        nb_refs = 8
        dimensions_definitions = {
            'possibility': "Question 1: In your opinion, is the answer possible? Does it make sense in some possible situation? 0: possible, in some conceivable situation / 1: impossible, or makes no sense",
            'obviousness': "Question 2: If the image was shown to 100 people and the question was asked, how many people do you think would give the answer? a number between 1 and 100",
            'correctness': "Question 3: Can you say that the answer is definitely factually correct (use Google if necessary)? 0: definitely correct, 1: can't say (matter of opinion, depends on the situation, the question does not have a 'right' answer, 2: definitely incorrect"
        }
        scale = "binary, likert and multiple choice"
        source_eval_sets = "NoVQA"
        annotators = "Expert annotators working at the Sorbonne University."
        additional_comments = "The annotation protocol is detailed on the appendix of this paper."
        sampled_from = "https://github.com/ThomasScialom/BEAMetrics/tree/main/beametrics"
        citation = """under review"""

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
        with open(path, 'r') as f:
            raw_data = json.load(f)

        d_data = {}
        # normalizing the annotations:
        for ex_id, ex in raw_data.items():
            d_data[ex_id] = ex
            d_data[ex_id]['source'] = ex['source_question']
            d_data[ex_id]['possibility'] = np.average([1-score for score in ex['possibility']])
            d_data[ex_id]['obviousness'] = np.average(ex['obviousness'])
            d_data[ex_id]['correctness'] = np.average([2 - score for score in ex['correctness']])

        return d_data


