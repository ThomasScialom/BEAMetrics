
## Installing The Dependencies
```
$ conda create --name beametrics python>=3.8
$ conda activate beametrics
```
**WARNING**: You need to install, before any package, correct version of [pytorch](https://pytorch.org/get-started/locally/#start-locally) linked to your cuda version.
```
(beametrics) $ conda install pytorch cudatoolkit=10.1 -c pytorch
```

# Computing the correlations:

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