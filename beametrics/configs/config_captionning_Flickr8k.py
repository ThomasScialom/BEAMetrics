import pickle
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES

class CaptioningFlickr8k(ConfigBase):
    def __init__(self):

        file_name = 'captioning_human_judgments.pkl'
        file_name_processed = 'processed.captioning.flickr8k'
        metric_names = _DEFAULT_METRIC_NAMES
        language = "en"
        task = "captioning"
        nb_refs = 5

        dimensions = ('score', )

        dimensions_definitions = {
            'score': "A score of 4 means that the caption describes the image perfectly (without any mistakes), "
            "a score of 3 that the caption almost describes the image (minor mistakes are allowed, e.g. in the number of entities), "
            "whereas a score of 2 indicates that the caption only describes some aspects of the image, but could not be used as its description, "
            "and a score of 1 indicates that the caption bears no relation to the image."
        }

        scale = "likert"

        sampled_from = "jair.org/index.php/jair/article/view/10833/25854"

        citation = """
                        @article{hodosh2013framing,
                        title={Framing image description as a ranking task: Data, models and evaluation metrics},
                        author={Hodosh, Micah and Young, Peter and Hockenmaier, Julia},
                        journal={Journal of Artificial Intelligence Research},
                        volume={47},
                        pages={853--899},
                        year={2013}}
                        """

        additional_comments = "The evaluation procedure is detailed Page 19 in https://www.jair.org/index.php/jair/article/view/10833/25854, in Section 4. Evaluation Procedures and Metrics for Image Description."

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
        keep_only = 'flickr8k'
        # Read the data
        with open(path, 'rb') as pickle_file:
            file_log = pickle.load(pickle_file)
            system = file_log[keep_only]

        d_data = dict()
        for i in range(len(system['score'])):
            ex = {
                'source': None,
                'references': system['reference'][i],
                'hypothesis': system['candidate'][i],
                'score': system['score'][i],
            }
            d_data[i] = ex

        return d_data