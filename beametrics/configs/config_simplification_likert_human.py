import numpy as np
import pandas as pd
import ast
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES, _DEFAULT_METRIC_NAMES_SRC

class SimplificationLikertHuman(ConfigBase):
    def __init__(self):

        file_name = 'simplification_absolute_ratings.csv'
        file_name_processed = 'processed.simplification.likert_human'
        metric_names = _DEFAULT_METRIC_NAMES + _DEFAULT_METRIC_NAMES_SRC
        metric_names = metric_names + ('sari', )

        language = "en"
        task = "simplification"
        nb_refs = 10

        dimensions = ('simplicity', 'meaning', 'fluency')

        dimensions_definitions = {
            'simplicity': "to what extent is the evaluated text easier to read and understand?",
            'meaning': "how well the evaluated text expresses the original meaning?",
            'fluency': "how fluent is the evaluated text?"
        }

        scale = "likert"

        sampled_from = "https://arxiv.org/pdf/2104.07560.pdf"

        citation = """@article{scialom2021rethinking,
                          title={Rethinking Automatic Evaluation in Sentence Simplification},
                          author={Scialom, Thomas and Martin, Louis and Staiano, Jacopo and de la Clergerie, {\'E}ric Villemonte and Sagot, Beno{\^\i}t},
                          journal={arXiv preprint arXiv:2104.07560},
                          year={2021}}"""

        additional_comments = "The evaluated texts are written by humans and evaluated on a likert scale from 1 to 5." \
                                   "The collection methodoloy is detailed in Appendix A of the paper. It follows ASSET: A Dataset for Tuning and Evaluation of the paper 'Sentence Simplification Models with Multiple Rewriting Transformations'."

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
        system_or_human = 'human'
        do_avg_raters = True
        normalise_rate = True

        df = pd.read_csv(path)

        d_worker_id = {}
        for i, row in df.iterrows():
            if row['worker_id'] not in d_worker_id:
                d_worker_id[row['worker_id']] = []
            d_worker_id[row['worker_id']].append(row['rating'])

        d_data = {}
        for i, row in df.iterrows():

            if row['simplification_type'] != system_or_human:
                continue

            ex_id = f"{row['sentence_id']}_{row['simplification']}"
            if ex_id not in d_data:
                d_data[ex_id] = {'source': row['source'],
                                 'hypothesis': row['simplification'],
                                 'references': ast.literal_eval(row['references']),
                                 }

            aspect = row['aspect']
            assert aspect in self.dimensions, f'Aspect {aspect} is not in the dimensions {self.dimensions}'
            if aspect not in d_data[ex_id]:
                d_data[ex_id][aspect] = []
            rate = row['rating']
            if normalise_rate:
                rate = (rate - np.average(d_worker_id[row['worker_id']])) / np.var(d_worker_id[row['worker_id']])
            d_data[ex_id][aspect].append(rate)

        d_annotators = {dim: [] for dim in self.dimensions}
        if do_avg_raters:
            for k in d_data.keys():
                for dim in self.dimensions:
                    d_annotators[dim].append(d_data[k][dim])
                    d_data[k][dim] = np.average(d_data[k][dim])

        """
        print('Krippendorff alpha: ')
        for dim in dimensions:
            kappa = krippendorff.alpha(np.array(d_annotators[dim]).transpose())
            print(dim, kappa)
        """

        return d_data