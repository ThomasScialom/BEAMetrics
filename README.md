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

## Download the data
All the dataset can be downloaded from [this zip file](https://drive.google.com/file/d/1axODMMTTeFUigkyC-JBoE8TgXsDA-CpJ/view?usp=sharing). It needs to be unzipped into the path data before running the correlations.

## Computing the correlations

Processing the files to a clean json with the metrics computed:
```
python beametrics/run_all.py
```
The optional argument *--dataset* allows to compute only on a specific dataset, e.g. *SummarizationCNNDM*. The list of the datasets and their corresponding configuration can be found in *configs/\_\_init\_\_*.



TODO: 
- refactor output files 
- mode per nb of ref
- clean readme with install and a printable result 
- allow to dynamically set new external metrics / remove one when running a config