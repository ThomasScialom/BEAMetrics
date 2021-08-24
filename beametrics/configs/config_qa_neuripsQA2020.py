import numpy as np
import json
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class NeurIPS2020openQA(ConfigBase):
    def __init__(self):

        file_name = 'NQ-open.efficientqa.test.1.1.jsonl'
        file_name_processed = 'processed.open_qa.neurips2020'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC

        name_dataset = 'NeurIPS_Efficient_QA_2020'
        short_name_dataset = 'QA'
        languages = ["en"]
        task = "QA"
        number_examples = 1800
        nb_refs = 1
        dimensions_definitions = {
            'consistency': "contains predictions that were determined to be definitely correct by annotators,",
            'plausibility': "contains preditions that were determined to be possibly correct given some interpretation of the question.",
        }
        scale = "binary"
        source_eval_sets = "EfficientQA competition at NeurIPS 2020"
        annotators = "In 2.3: three separate raters"
        additional_comments = """- Based on our human evaluation, annotations on ambiguity have low agreement rate (61.3%, Cohen’s κ = 22.6), and predictions with the same level of plausibility are often marked as “definitely correct” or “possibly correct” by different human raters. 
        - The annotation and Json keys are described in https://github.com/google-research-datasets/natural-questions/tree/master/nq_open. 
        - Discrepancy between automatic eval and human eval Human raters find 13% and 17% of the predictions that do not match the reference answers to be definitely correct or possibly correct, respectively, overall increasing the accuracy of the systems. Most systems showed 17–25% and 41–54% improvement in accuracy when using definitely correct and possibly correct human evaluation respectively, compared to automatic evaluation metric which only consider exact string match to existing reference answers. An exception is NAVER RDR, which achieves significantly larger improvements (32% and 71%, respectively). We also found that when the gap in automatic measure between systems is marginal (around or smaller than 1%), human evaluation may change the rankings between the models."""
        sampled_from = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.efficientqa.test.1.1.jsonl"
        citation = """@article{min2021neurips,
        title={NeurIPS 2020 EfficientQA competition: Systems, analyses and lessons learned},
        author={Min, Sewon and Boyd-Graber, Jordan and Alberti, Chris and Chen, Danqi and Choi, Eunsol and Collins, Michael and Guu, Kelvin and Hajishirzi, Hannaneh and Lee, Kenton and Palomaki, Jennimaria and others},
        journal={arXiv preprint arXiv:2101.00133},
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
        items = []
        with open(path, "r") as f:
            while True:
                l = f.readline()
                if len(l) == 0:
                    break
                items.append(json.loads(l))

        map_annotation_2_labels = {
            'def_correct_predictions': {'consistency': 1, 'plausibility': 1},
            'poss_correct_predictions': {'consistency': 1, 'plausibility': 0},
            'def_incorrect_predictions': {'consistency': 0, 'plausibility': 0},
        }

        d_data = {}
        for i, item in enumerate(items):

            for annotation_type in map_annotation_2_labels.keys():
                for hyp_idx, hyp in enumerate(item[annotation_type]):
                    d_data_key = f'{i}.{annotation_type}.{hyp_idx}'
                    d_data[d_data_key] = {
                        'hypothesis': hyp,
                        'references': item['answer'],
                        'source': item['question'],
                    }
                    if len(item['answer'])>15:
                        temp=1
                    for annotation, label in map_annotation_2_labels[annotation_type].items():
                        d_data[d_data_key][annotation] = label

        return d_data
