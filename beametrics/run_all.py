from scipy.stats import pearsonr, kendalltau
from beametrics.utils import component_logger
import argparse
from beametrics.configs import D_ALL_DATASETS

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default= None,
        help=f"The dataset you want to compute the correlation. "
             f"It has to be in: \n {D_ALL_DATASETS.keys()} \n"
             f"If None, all the datasets will be computed."
    )

    parser.add_argument(
        '--path_data',
        type=str,
        default='data',
        help=f"The path where the dataset file is stored. "
             f"By default BEEval/data."
    )

    parser.add_argument(
        '--reload_cache',
        default=False,
        type=bool,
        help=f"If True the processed files will be overwritten."
    )

    args = parser.parse_args()

    configs = D_ALL_DATASETS.keys()
    if args.dataset is not None:
        assert args.dataset in D_ALL_DATASETS
        configs =[args.dataset]

    for config_name in configs:
        component_logger.info(f'________________________________________________\n'
                              f'Config {config_name} loaded. Computing the pipeline.')
        config = D_ALL_DATASETS[config_name]()
        config.generate_data_card()

        for f in [pearsonr, kendalltau]:
            config.pipeline(
                path_data=args.path_data,
                reload_cache=args.reload_cache,
                correl_function=f
            )


if __name__ == "__main__":
    main()


