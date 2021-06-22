from typing import List, Tuple, Dict, Callable
import os
import json
from scipy.stats import pearsonr

from beeval.metrics.metric_reporter import MetricReporter
from beeval.utils import component_logger

class ConfigBase:
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
        additional_comments: str = None
    ):

        self.file_name = file_name
        self.file_name_processed = file_name_processed
        self.language = language
        self.task = task
        self.nb_refs = nb_refs
        self.dimensions = dimensions
        self.dimensions_definitions = dimensions_definitions
        self.scale = scale
        self.sampled_from = sampled_from
        self.citation = citation
        self.additional_comments = additional_comments

        self.metric_names = metric_names
        self.metric_reporter = MetricReporter(
            metric_names=metric_names
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
        raise(NotImplemented)

    def fill_rank(self, *args, **kwargs):
        """
        Usage: to normalise metrics for pairwise evaluation. No use for likert rating.
        """
        pass

    def pipeline(
        self,
        path_data: str ='data',
        use_cache: bool =True,
        correl_function: Callable =pearsonr
    ):
        """
        The all pipeline that will
            1) open a dataset file
            2) format it
            3) compute the metrics
            4) calculate the metrics correlations
        :param path_data: the path of the raw evaluation dataset
        :param use_cache: if True will use the stored results for the metrics already computed
        :return: the metrics correlations for the dataset
        """

        path = os.path.join(path_data, self.file_name)
        assert os.path.exists(path), f"Path {path} does not exist."
        path_processed_data = os.path.join(path_data, self.file_name_processed)
        if not use_cache or not os.path.exists(path_processed_data):
            # Load the raw data
            d_data = self.format_file(path)
        else:
            # Load the processed data
            with open(path_processed_data, "r") as f_r:
                d_data = json.loads(f_r.read())

        # Fill the metrics that has not been computed on d_data yet.
        for ex_id in d_data: break
        to_do_metrics = self.metric_reporter.get_missing_metrics(
            a_dict=d_data[ex_id],
            metric_names=self.metric_names
        )

        if len(to_do_metrics) > 0:
            component_logger.info(
                f"Data loaded. Now computing the following metrics: \n"
                f"{' '.join(to_do_metrics)}."
            )
            self.add_missing_metrics(d_data, to_do_metrics)

        # Serialised the processed data
        json_data = json.dumps(d_data)
        with open(path_processed_data, "w") as f_w:
            f_w.write(json_data)

        # Run the correlations
        component_logger.info('Processing done. Computing the correlations.')
        self.correlation(
            d_data=d_data,
            correl_function=correl_function,
            path_data=path_data
        )

    def add_missing_metrics(
        self,
        d_data: Dict,
        to_do_metrics: List[str]
    ):

        predictions = self.get_val(d_data, 'hypothesis')
        assert predictions[0] is not None

        sources = self.get_val(d_data, 'source')
        if sources[0] == None: sources = None

        list_references = self.get_val(d_data, 'references')
        if list_references[0][0] == None: list_references = None

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

        # Add all the metrics to d_data
        for i, ex_id in enumerate(d_data):
            for metric_name in sub_list_metrics:
                d_data[ex_id][metric_name] = d_res[metric_name][i]

    def correlation(
        self,
        d_data: Dict,
        correl_function: Callable,
        path_data: str,
    ):

        dim_1s = self.dimensions
        dim_2s = tuple(self.metric_reporter.get_sub_metric_names(self.metric_names))

        # Format if needed (for pairwise rating)
        self.fill_rank(d_data=d_data,
                       dim_1s=dim_1s,
                       dim_2s=dim_2s
                       )

        # Computing the correlations
        d_scores, d_pvalue = self.compute_correl(d_data=d_data,
                                                 dim_1s=dim_1s,
                                                 dim_2s=dim_2s,
                                                 correl_function=correl_function
                                                 )

        # Serializing the correlations
        file_correlations = self.file_name_processed.replace('processed', 'correlations')
        path_correlations = os.path.join(path_data, file_correlations)
        d_correls = {'d_scores': d_scores, 'd_pvalue': d_pvalue}
        with open(path_correlations, "w") as f_w:
            json.dump(d_correls, f_w, indent=2)
        component_logger.info(d_correls)

    def compute_correl(
        self,
        d_data: Dict,
        dim_1s: List[str],
        dim_2s: List[str],
        correl_function: Callable
    ):
        d_scores, d_pval = {}, {}
        for dim_1 in dim_1s:
            d_scores[dim_1] = dict()
            d_pval[dim_1] = dict()
            for dim_2 in dim_2s:
                coef, p_value = correl_function(
                    self.get_val(d_data, dim_1),
                    self.get_val(d_data, dim_2)
                )
                d_scores[dim_1][dim_2] = round(100 * coef, 1)
                d_pval[dim_1][dim_2] = format(p_value, ".3E")

        return d_scores, d_pval
