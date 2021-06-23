import pickle
from beeval.configs.config_base import ConfigBase
from beeval.metrics.metric_reporter import _DEFAULT_METRIC_NAMES


class CaptioningComposite(ConfigBase):
    def __init__(self):

        file_name = 'captioning_human_judgments.pkl'
        file_name_processed = 'processed.captioning.composite'
        metric_names = _DEFAULT_METRIC_NAMES
        language = "en"
        task = "captioning"
        nb_refs = 5

        dimensions = ('score', )

        dimensions_definitions = {'score': "relevance between each candidate caption-image pair with 5 reference captions",
        }

        scale = "likert"

        sampled_from = "https://imagesdg.wordpress.com/image-to-scene-description-graph/"

        citation = """
                        @article{aditya2015images,
                        title={From images to sentences through scene description graphs using commonsense reasoning and knowledge},
                        author={Aditya, Somak and Yang, Yezhou and Baral, Chitta and Fermuller, Cornelia and Aloimonos, Yiannis},
                        journal={arXiv preprint arXiv:1511.03292},
                        year={2015}}
                        """

        additional_comments = "The likert scale is 1 to 5."

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
        keep_only = 'composite'
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