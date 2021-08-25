# BEAMetrics: Benchmark to Evaluate Automatic Metrics in Natural Language Generation

## Installing The Dependencies
```
$ conda create --name beametrics python>=3.8
$ conda activate beametrics
```
**WARNING**: You need to install, before any package, correct version of [pytorch](https://pytorch.org/get-started/locally/#start-locally) linked to your cuda version.
```
(beametrics) $ conda install pytorch cudatoolkit=10.1 -c pytorch
```

Install BEAMetrics:
```
(beametrics) $ cd BEAMetrics
(beametrics) $ pip install -e .
```

Install Nubia metric (not on PyPI, 16/08/2021):
```
(beametrics) git clone https://github.com/wl-research/nubia.git
(beametrics) pip install -r requirements.txt
```
Alternatively, you can remove nubia from `_DEFAULT_METRIC_NAMES` in `metrics.metric_reporter`.

## Reproducing the results

First you need to get the processed files, which include the metric scores. You can do that either by simply downloading the processed data (see [Section Download Data](#download-the-data)), or by re-computing the scores (see [Section Compute Correlations](#computing-the-correlations)). 

Then, the first bloc in the notebook `visualize.ipynb` allows to get all the tables from the paper (and also to generate the latex code in `data/correlation`).


## Download the data
All the dataset can be downloaded from [this zip file](https://drive.google.com/file/d/1ILzn7tRZqBUYf9yb3IskyvLL_eo53dOA/view?usp=sharing). It needs to be unzipped into the path data before running the correlations.
```
unzip data.zip
```

The `data` folder contains:
- a subfolder `raw` containing all the original dataset
- a subfolder `processed` containing all the dataset processed in a unified format
- a subfolder `correlation` containing all the final correlation results, and the main tables of the paper
- a subfolder `datacards` containing all the data cards

## Computing the correlations

Processing the files to a clean json with the metrics computed:
```
python beametrics/run_all.py
```
 
The optional argument `--dataset` allows to compute only on a specific dataset, e.g.:

 `python run_all.py --dataset SummarizationCNNDM`. 
 
The list of the datasets and their corresponding configuration can be found in `configs/__init__`.

When finished, you can print the final table as in the paper, see the notebook `visualize.ipynb`.

## Data Cards:

For each dataset, a data card is available in the [datacard folder](https://github.com/ThomasScialom/BEAMetrics/tree/main/data/datacards). The cards are automatically generated when running `run_all.py`, by filling the template with the dataset configuration as detailed bellow, in *[Adding a new dataset](#adding-a-new-dataset)*. 

## Adding a new dataset:

In `configs/`, you need to create a new `.py` file that inherites from `ConfigBase` (in `configs/co'nfig_base.py`). 
You are expected to fill the mandatory fields that allow to run the code and fill the data card template:
- `file_name`: the file name located in `data/raw`
- `file_name_processed`: the file name once processed and formated
- `metric_names`: you can pass _DEFAULT_METRIC_NAMES by default or customize it, e.g. `metric_names = metric_names + ('sari',)` where `sari` corresponds to a valid metric (see the next section)
- `name_dataset`: the name of the dataset as it was published
- `short_name_dataset`: few letters that will be used to name the dataset in the final table report
- `languages`: the languages of the dataset (e.g. `[en]` or `[en, fr]`)
- `task`: e.g. `'simplification'`, `'data2text`
- `number_examples`: the total number of evaluated texts
- `nb_refs`: the number of references available in the dataset
- `dimensions_definitions`: the evaluated dimensions and their corresponding definition e.g. `{'fluency: 'How fluent is the text?'}`
- `scale`: the scale used during the evaluation, as defined in the protocol
- `source_eval_sets`: the dataset from which the source were collected to generate the evaluated examples
- `annotators`: some information about who were the annotators
- `sampled_from`: the URL where was released the evaluation dataset
- `citation`: the citation of the paper where the dataset was released

Your class needs its custom method `format_file`. The function takes as input the dataset's `file_name` and return a dictionary `d_data`.
The format for `d_data` has to be the same for all the datasets:

```
d_data = {
    key_1: {
        'source': "a_source", 
        'hypothesis': "an_hypothesis",
        'references': ["ref_1", "ref_2", ...],
        'dim_1': float(a_score),
        'dim_2': float(an_other_score),
    },
    ...
    key_n: {
        ...
    }
}
```
where `'key_1'` and `'key_n'` are the keys for the first and n<sup>th</sup> example, `dim_1` and `dim_2` dimensions corresponding to `self.dimensions`.

Finally, you need to add your dataset to the dictionary `D_ALL_DATASETS` located in `config/__init__`.

## Adding a new metric:

First, create a class inheriting from `metrics/metrics/MetricBase`. Then, simply add it to the dictionary `_D_METRICS` in `metrics/__init__`.

For the metric to be computed by default, its name has to be added to either
- `_DEFAULT_METRIC_NAMES`: metrics computed on each dataset
- `_DEFAULT_METRIC_NAMES_SRC`: metrics computed on dataset that have a text format for their source (are excluded for now image captioning and data2text).
These two tuples are located in `metrics/metric_reported`. 

Alternatively, you can add the metric to a specific configuration by adding it to the attribute `metric_names` in the config.

