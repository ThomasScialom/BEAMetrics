import json
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class SimplificationMUSS(ConfigBase):
    def __init__(self):

        file_name = 'muss.json'
        file_name_processed = 'processed.simplification.MUSS'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC
        metric_names = metric_names + ('sari',)

        name_dataset = 'MUSS'
        short_name_dataset = 'MUS'
        languages = ["en", 'fr', 'es', ]
        task = "simplification"
        number_examples = 150
        nb_refs = 1
        dimensions_definitions = {
            'simplicity': "is the output simpler than the original sentence?",
            'adequacy': "Adequacy: to what extent is the meaning expressed in the original sentence preserved in the output?",
            'fluency': "is the output grammatical and well formed?"
        }
        scale = "likert 1-4"
        source_eval_sets = "MUSS"
        annotators = "We recruit volunteer native speakers for each language (5 in English, 2 in French, and 2 in Spanish)"
        additional_comments = """
        Please sign up [here](https://newsela.com/data/) to access the Newsela data.
        
        For simplicity, please consider only simplification operations between the source and simplification, i.e. modifications that actually make the sentence simpler to read and understand, based on your own judgement.

        Scale:
        4 = Entirely simpler (It seems like all possible simplification operations were conducted, it seems like it is impossible to make the sentence simpler)
        3 = Much simpler (3 or more simplification operations of different types)
        2 = Simpler (2 or 3 simple simplifications or one very relevant simplification)
        1 = Slightly simpler (e.g. only one simple simplification operation such as removing a few words)
        0 = Not simpler or more complicated
        
        Simplification operations  by order of complexity include but are not limited to:
        Removing an adjective or details in parenthesis (unimportant information).
        Shortening a sentence to make it easier to read by removing clauses.
        Replacing a complex word with a simpler one.
        Changing complicated tenses (e.g. past perfect) to simpler ones (e.g. present).
        Splitting long sentences into multiple.
        Reordering the information of a sentence to make it more natural.
"""
        sampled_from = "https://arxiv.org/pdf/2005.00352.pdf"
        citation = """@article{martin2020muss,
        title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
        author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
        journal={arXiv preprint arXiv:2005.00352},
        year={2020}
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
        path
    ):
        with open(path, 'r') as f:
            d_data = json.load(f)

        return d_data