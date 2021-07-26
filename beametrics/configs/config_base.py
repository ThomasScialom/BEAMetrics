from typing import List, Tuple, Dict, Callable, Any
import os
import json
from scipy.stats import pearsonr
import pprint

from beametrics.metrics.metric_reporter import MetricReporter
from beametrics.utils import component_logger

class ConfigBase:
    """
    Config class from which any new dataset inherits.
        Its pipeline methods allows to compute the correlation for the list of metrics in metric_names
        given a raw file file_name.
        The method format_file has to be implemented individually for each config: it converts any
        specific format in file_name to d_data, a standardised dict with:

        d_data:
        {
            key_example_1:
            {
                hypothesis -> the prediction
                source -> the source
                references -> the list of N references with N = self.nb_refs
                dim_1 -> the normalised averaged score for the hypothesis given by the annotators wrt dim_1
                         where dim_1 is one of the evaluated dimension in self.dimensions.
                other non standardised information
            }
        }
    """

    def __init__(
        self,
        file_name: str,
        file_name_processed: str,
        metric_names: Tuple[str],
        language: str,
        task: str,
        nb_refs: int,
        dimensions: Tuple[str],
        dimensions_definitions: Dict[str, str],
        scale: str,
        sampled_from: str,
        citation: str,
        additional_comments: str = None,
        list_nb_ref: Tuple = None
    ):
        """
        Arguments:
            file_name (str):
                The raw file for the human evaluation (should be in data/raw/).
            file_name_processed (str):
                The name of the standardized json file (should be in data/processed/).
            metric_names (Tuple[str]):
                The list of metrics to compute on this task. If not specified it will be computed
                on the default ones.
            language (str):
                The language of the dataset (e.g. en, fr, zh).
            task (str):
                The task of the dataset (e.g. summarization, data2text).
            nb_refs (int):
                The number of human references available in the dataset.
            list_nb_ref (Tuple/List/Range):
                The different number of references we want to compute the correlations.
                If not specify, the pipeline will be computed on [1, self.nb_refs].
            dimensions (Tuple[str]):
                The evaluated dimensions during the human evaluation,
                and on which the correlation is computed.
            dimensions_definitions (Dict[str,str]:
                A dictionary where each dimension in self.dimensions is a key. The corresponding value
                is the definiton of the dimension according to the original paper.
            scale (str):

            sampled_from (str):
                The URL of the dataset.
            citation (str):
                The official citation for the dataset.
            additional_comments (str):
                Any relevant information, specificities about the dataset.
        """

        self.file_name = file_name
        self.file_name_processed = file_name_processed
        self.language = language
        self.task = task
        self.nb_refs = nb_refs
        self.list_nb_ref = list_nb_ref
        self.dimensions = dimensions
        self.dimensions_definitions = dimensions_definitions
        self.scale = scale
        self.sampled_from = sampled_from
        self.citation = citation
        self.additional_comments = additional_comments

        self.metric_names = metric_names
        self.metric_reporter = MetricReporter(
            metric_names=metric_names,
            task=self.task,
            lang=self.language
        )
        if self.metric_names is None:
            self.metric_names = self.metric_reporter.metric_names

        self.get_val = lambda d_data, dim: \
            [ex[dim] for id_ex, ex in d_data.items()]

    def format_file(self):
        """
        Usage: a formating function that corresponds to the specific dataset.
        :return: d_data, a unified format among all the datasets
        """
        raise NotImplementedError

    def fill_rank(self, *args, **kwargs):
        """
        Usage: To normalise the metrics in case of a pairwise evaluation.
        No use for likert rating.
        """
        pass

    def pipeline(
        self,
        path_data: str ='data',
        reload_cache: bool = False,
        correl_function: Callable = pearsonr
    ):
        """
        The all pipeline that will
            1) open a dataset file
            2) format it
            3) compute the metrics
            4) calculate the metrics correlations
        Arguments:
            path_data (str):
                The path of the raw evaluation dataset.
            reload_cache (bool):
                If False will use the stored results for the metrics already computed.
        """

        path_raw_data = os.path.join(path_data, 'raw', self.file_name)
        assert os.path.exists(path_raw_data), f"Path {path_raw_data} does not exist."

        path_processed_data = os.path.join(path_data, 'processed', self.file_name_processed)
        if reload_cache or not os.path.exists(path_processed_data):
            # Load the raw data
            d_data = self.format_file(path_raw_data)
        else:
            # Load the processed data
            with open(path_processed_data, "r") as f_r:
                d_data = json.loads(f_r.read())

        # Fill the metrics that has not been computed on d_data yet.
        ref_keys = []
        list_nb_ref = [1, self.nb_refs] if self.list_nb_ref is None else self.list_nb_ref
        for nb_ref in list_nb_ref:
            ref_key = f'ref_{nb_ref}'
            ref_keys.append(ref_key)

            for ex_id in d_data: break
            if ref_key not in d_data[ex_id]:
                to_do_metrics = self.metric_names
            else:
                to_do_metrics = self.metric_reporter.get_missing_metrics(
                    a_dict=d_data[ex_id][ref_key],
                    metric_names=self.metric_names,
                )

            if len(to_do_metrics) > 0:
                component_logger.info(
                    f"Data loaded. Now computing the following metrics: \n"
                    f"{' '.join(to_do_metrics)}."
                )
                d_res = self.get_missing_metrics(
                    d_data=d_data,
                    to_do_metrics=to_do_metrics,
                    nb_ref=nb_ref
                )

                # Add all the metrics to d_data
                for i, ex_id in enumerate(d_data):
                    if ref_key not in d_data[ex_id]:
                        d_data[ex_id][ref_key] = dict()
                    for metric_name in d_res.keys():
                        d_data[ex_id][ref_key][metric_name] = d_res[metric_name][i]

                # Serialised the processed data
                json_data = json.dumps(d_data)
                with open(path_processed_data, "w") as f_w:
                    f_w.write(json_data)

        # Run the correlations
        component_logger.info('Processing done. Computing the correlations.')
        self.correlation(
            d_data=d_data,
            correl_function=correl_function,
            path_data=path_data,
            ref_keys=ref_keys
        )

    def get_missing_metrics(
        self,
        d_data: Dict,
        to_do_metrics: List[str],
        nb_ref: int
    ):

        predictions = self.get_val(d_data, 'hypothesis')
        assert predictions[0] is not None

        sources = self.get_val(d_data, 'source')
        if sources[0] == None:
            sources = None

        list_references = self.get_val(d_data, 'references')
        # We consider that taking the N first references is not less random than to sample.
        # In addition, it simplifies the reproductibility.
        list_references = [references[:nb_ref] for references in list_references]
        if nb_ref == 0:
            list_references = None

        # Compute the metrics
        d_res = self.metric_reporter.compute(
            predictions = predictions,
            sources = sources,
            list_references = list_references,
            metric_names=to_do_metrics
        )

        sub_list_metrics = self.metric_reporter.get_sub_metric_names(to_do_metrics)
        assert any(
            [len(d_data) == len(d_res[metric_name]) for metric_name in sub_list_metrics]
        )

        return d_res

    def correlation(
        self,
        d_data: Dict,
        correl_function: Callable,
        path_data: str,
        ref_keys: List[str]
    ):
        dim_humans = self.dimensions
        dim_metrics = tuple(self.metric_reporter.get_sub_metric_names(self.metric_names))

        d_correlations = dict()
        for ref_key in ref_keys:
            d_correlations[ref_key] = dict()
            # Format if needed (for pairwise rating)
            self.fill_rank(
                d_data=d_data,
                ref_key=ref_key,
                dim_humans=dim_humans,
                dim_metrics=dim_metrics,
            )

            # Computing the correlations
            d_scores, d_pvalue = self.compute_correl(
                d_data=d_data,
                ref_key=ref_key,
                dim_humans=dim_humans,
                dim_metrics=dim_metrics,
                correl_function=correl_function
            )
            d_correlations[ref_key]['d_scores'] = d_scores
            d_correlations[ref_key]['d_pvalue'] = d_pvalue

        # Serializing the correlations
        path_correlations = os.path.join(path_data, 'correlation', self.file_name_processed)
        with open(path_correlations, "w") as f_w:
            json.dump(d_correlations, f_w, indent=2)
        component_logger.info(d_correlations) #pprint.pprint

    def compute_correl(
        self,
        d_data: Dict,
        ref_key: str,
        dim_humans: List[str],
        dim_metrics: List[str],
        correl_function: Callable
    ) -> (Dict[str,Dict[str, float]], Dict[str,Dict[str, float]]):
        d_scores, d_pval = {}, {}
        for dim_h in dim_humans:
            d_scores[dim_h] = dict()
            d_pval[dim_h] = dict()
            for dim_m in dim_metrics:
                coef, p_value = correl_function(
                    self.get_val(d_data, dim_h),
                    [ex[ref_key][dim_m] for id_ex, ex in d_data.items()]
                )
                d_scores[dim_h][dim_m] = round(100 * coef, 1)
                d_pval[dim_h][dim_m] = format(p_value, ".3E")

        return d_scores, d_pval
