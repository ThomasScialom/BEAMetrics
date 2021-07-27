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

## Download the data
All the dataset can be downloaded from [this zip file](https://drive.google.com/file/d/1axODMMTTeFUigkyC-JBoE8TgXsDA-CpJ/view?usp=sharing). It needs to be unzipped into the path data before running the correlations.
```
cd data/raw
unzip data.zip
```
## Computing the correlations

Processing the files to a clean json with the metrics computed:
```
python beametrics/run_all.py
```

When finished, you can print the final table as in the paper: 
# todo
 
The optional argument `--dataset` allows to compute only on a specific dataset, e.g.:

 `python run_all.py --dataset SummarizationCNNDM`. 
 
The list of the datasets and their corresponding configuration can be found in `configs/__init__`.

## Adding a new dataset:

In `configs/`, you need to create a new `.py` file that inherites from `ConfigBase` (in `configs/config_base.py`). 
You are expected to fill the mandatory fields at least:
- file_name: the file name located in `data/raw`,
- file_name_processed: the file name once processed and formated,
- metric_names: you can pass _DEFAULT_METRIC_NAMES by default or customize it, e.g. `metric_names = metric_names + ('sari',)` where `sari` corresponds to a valid metric (see the next section),
- language: the language of the dataset (e.g. ``'en'`, `'multi'`)
- task: e.g. `'simplification'`, `'data2text`
- nb_refs: the number of references available in the dataset
- dimensions: the evaluated dimensions e.g. `('simplicity', 'meaning', 'fluency')`

But more information is better, for instance the exact definition of the dimensions or the citation.  

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

First, create a class inheriting from `metrics/metrics/MetricBase`.