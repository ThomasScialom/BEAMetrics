import numpy as np
import json
from beametrics.configs.config_base import ConfigBase
from beametrics.metrics.metric_reporter import _DEFAULT_METRIC_NAMES

class SummarizationCNNDM(ConfigBase):
    def __init__(self):

        file_name = 'summeval.model_annotations.aligned.paired.jsonl'
        file_name_processed = 'processed.summarization.cnndm'
        metric_names = _DEFAULT_METRIC_NAMES

        language = "en"
        task = "summarization"
        nb_refs = 11

        dimensions = ('coherence', 'consistency', 'fluency', 'relevance')

        dimensions_definitions = {
            'coherence': "‘the summary should be well-structured and well-organized. The summary should not just be a heap of related information, should build from sentence to sentence to a coherent body of information about a topic.",
            'consistency': "the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.",
            'fluency': "the quality of individual sentences. Drawing again from the DUC quality guidelines, sentences in the summary ‘‘should have no formatting problems, capitalization errors or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read.’’",
            'relevance': "selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries that contained redundancies and excess information."
        }

        scale = "likert"

        sampled_from = "https://arxiv.org/abs/2007.12626"

        citation = """@article{fabbri2020summeval,
                      title={SummEval: Re-evaluating Summarization Evaluation},
                      author={Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
                      journal={arXiv preprint arXiv:2007.12626},
                      year={2020}
                    }
        """

        additional_comments = "- The authors release two set of annotations: " \
                              "from experts and turkers; we keep only the expert one." \
                              "- The present file is derived from the original one released by the authors," \
                              "it contains in addition a key 'text' that maps the example to the source article."

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

        d_data = {}
        for i, item in enumerate(items):
            d_data[i] = dict()

            d_data[i]['hypothesis'] = item['decoded']
            d_data[i]['references'] = item['references']
            d_data[i]['source'] = item['text']

            for dim in self.dimensions:
                d_data[i][dim] = np.average([d_rate[dim] for d_rate in item['expert_annotations']])

            d_data[i]['id'] = item['id']
            d_data[i]['model_id'] = item['model_id']
            d_data[i]['filepath'] = item['filepath']

        return d_data
