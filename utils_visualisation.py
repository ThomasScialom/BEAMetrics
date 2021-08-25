import json
import pandas as pd
import re

import matplotlib.pyplot as plt
from matplotlib import colors

def apply_heatmap(s, cmap='PuBu', low=0, high=0):
    # Pass the columns from Dataframe A
    a = s.copy()
    a = [abs(v) if isinstance(v, float) else 0 for v in a.values]
    rng = max(a) - min(a)
    norm = colors.Normalize(min(a) - (rng * low), max(a) + (rng * high))

    normed = norm(a)

    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def get_midrule_idx(df):
    midrule_idx = []
    for i in range(1, len(df.index)):
        if df.index[i][0] != df.index[i - 1][0]:
            midrule_idx.append(i + len(midrule_idx))
    return midrule_idx


def get_indexes_to_remove(df, strs_to_remove=[]):
    return [
        (index[0], index[1])
        for string in strs_to_remove
        for i, index in enumerate(df.index)
        if index[1] == string
    ]


def get_complete_df(d_scores):
    dfs = []
    dfs_indexes = []

    # header
    df = pd.DataFrame(d_scores['header'])
    index = [('#Ref', h) for h in df.index.to_list()]
    df.index = index

    dfs.append(df)
    dfs_indexes += index

    # blocs
    for nb_ref_name in ['Max', "1", "0"]:
        dict_score = d_scores[nb_ref_name]
        df = pd.DataFrame(dict_score)
        index = [(nb_ref_name, m) for m in list(list(dict_score.values())[0].keys())]
        df.index = index

        dfs.append(df)
        dfs_indexes += index

    df = pd.concat(dfs)

    dfs_indexes = pd.MultiIndex.from_tuples(dfs_indexes)
    df = df.reindex(dfs_indexes)

    return df


def merge_wmt_cols(df):
    wmt_cols = [col for col in df.columns if 'WMT' in col]

    if wmt_cols:
        vals = []
        for idx in df.index:
            val = round(sum(df[wmt_cols].loc[idx]) / len(wmt_cols), 1) if idx[0] != '#Ref' else df[wmt_cols].loc[idx][0]
            vals.append(val)
        df['WMT<DaRR>'] = vals

        for c in wmt_cols:
            df.drop(c, axis=1, inplace=True)

        df = df[['WMT<DaRR>'] + [col for col in df.columns if col != 'WMT<DaRR>']]

    return df


def format_latex_table(latex_table, midrule_idx, nb_col_values):
    def format_numbers(string):
        i = 0
        while i < len(string):

            if all([c in '0123456789' for c in string[i:i + 1]]) and string[i + 1] == '.':
                if all([c in '0123456789' for c in string[i + 2:i + 4]]):
                    if all([c == '0' for c in string[i + 4:i + 8]]):
                        string = string[:i + 3] + string[i + 8:]
                        i -= 5

            i += 1
        return string

    lines = latex_table.splitlines()

    # center columns for the values
    lines[0] = lines[0].replace('l' * (nb_col_values + 2), 'll' + 'c' * nb_col_values)

    # remove the dimension from the col name
    lines[1] = re.sub(r'<.+?>', '', lines[1])

    # header in white
    for color in re.findall('\[HTML\]{(.*?)}', lines[2]):
        lines[2] = lines[2].replace(color, 'FFFFFF')

    if 'dimension' in lines[3]:
        for color in re.findall('\[HTML\]{(.*?)}', lines[3]):
            lines[3] = lines[3].replace(color, 'FFFFFF')

    # uppercase for task and dimension
    lines[2] = lines[2].replace('task', 'Task:')
    lines[3] = lines[3].replace('dimension', 'Dim:')

    # format the numbers with 2 digits after comma
    lines = [format_numbers(l) for i, l in enumerate(lines)]

    # add midrules between the blocs
    for idx in midrule_idx:
        lines = lines[:idx + 2] + ['\\midrule'] + lines[idx + 2:]

    # add top and bottom rules
    lines = lines[:1] + ['\\toprule'] + lines[1:-1] + ['\\bottomrule'] + lines[-1:]

    latex_table = '\n'.join(lines)
    latex_table = latex_table.replace('#', '\#')

    return latex_table


def make_dataframe_from_dict(path_correlations, d_scores_global, key, correl_func, str_to_removes, cmap='PuBu'):
    d_scores = d_scores_global[key]
    df = get_complete_df(d_scores)
    if key == 'Correctness':
        str_to_removes += ['dimension']
        df = merge_wmt_cols(df)
        df['Avg'] = ['All', 'All'] + [
            round(sum(df.replace('-', 0)[df.columns].loc[idx].values) / len(df[df.columns]), 1) for idx in df.index[2:]]

    df.loc[('#Ref', 'dimension')] = [v[:3].capitalize() for v in df.loc[('#Ref', 'dimension')].values]
    remove_indexes = get_indexes_to_remove(df, str_to_removes)
    df = df.drop(remove_indexes)
    midrule_idx = get_midrule_idx(df)

    s = df.style.apply(apply_heatmap, cmap=cmap)
    latex_table = s.to_latex(convert_css=True)
    latex_table = format_latex_table(latex_table, midrule_idx, len(df.columns))

    with open(path_correlations + f'final_table_{key}_{correl_func}_values.txt', 'w') as f_w:
        f_w.write(latex_table)

    return s


def get_d_scores_global(D_ALL_DATASETS, d_short_task, correl_function, path_correlations):
    d_scores_global = {
        'Correctness': {'header': {}, 'Max': {}, '1': {}, "0": {}},
        'other': {'header': {}, 'Max': {}, '1': {}, "0": {}}
    }
    indexes = []

    for dataset in D_ALL_DATASETS:

        config = D_ALL_DATASETS[dataset]['class']()
        file_name_processed = config.file_name_processed
        with open(f'{path_correlations}{file_name_processed}.{correl_function}', 'r') as f:
            correl_file = json.load(f)

        for name_dim_print, name_dim_dataset in D_ALL_DATASETS[dataset]['map_dim'].items():

            name_dim_glob = name_dim_print if name_dim_print in d_scores_global else 'other'

            name_col = f'{config.short_name_dataset}<{name_dim_dataset}>'
            if name_col not in d_scores_global[name_dim_glob]['header']:
                d_scores_global[name_dim_glob]['header'][name_col] = {
                    'task': d_short_task[config.task], 'dimension': name_dim_print
                }

            # ______________________________________________________ Max Ref  ______________________________________________________

            key_nb_ref = f'ref_{config.nb_refs}'
            nb_ref_name = 'Max'
            d_scores_global[name_dim_glob][nb_ref_name][name_col] = {
                'ROUGE-1': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['rouge1'],
                'ROUGE-2': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['rouge2'],
                'ROUGE-L': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['rougeL'],
                'BLEU': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['sacrebleu'],
                'METEOR': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['meteor'],
                # 'SARI': 'Nan',
                'BERTScore P': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bertscore_precision'],
                'BERTScore R': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bertscore_recall'],
                'BERTScore F1': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bertscore_f1'],
                'BLEURT': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bleurt'],
                'Nubia': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['nubia_score'],
                'Nubia (irrelevancy)': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['nubia_irrelevancy'],
            }

            indexes.append(
                [(nb_ref_name, m) for m in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore P',
                                            'BERTScore R', 'BERTScore F1', 'BLEURT', 'Nubia', 'Nubia (irrelevancy)']]
            )
            # ______________________________________________________ Ref 1 ______________________________________________________

            key_nb_ref = 'ref_1'
            nb_ref_name = '1'
            d_scores_global[name_dim_glob][nb_ref_name][name_col] = {
                'ROUGE-1': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['rouge1'],
                'ROUGE-2': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['rouge2'],
                'ROUGE-L': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['rougeL'],
                'BLEU': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['sacrebleu'],
                'METEOR': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['meteor'],
                # 'SARI': 'Nan',
                'BERTScore P': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bertscore_precision'],
                'BERTScore R': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bertscore_recall'],
                'BERTScore F1': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bertscore_f1'],
                'BLEURT': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['bleurt'],
                'Nubia': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['nubia_score'],
                'Nubia (irrelevancy)': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['nubia_irrelevancy'],
            }
            indexes.append(
                [(nb_ref_name, m) for m in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore P',
                                            'BERTScore R', 'BERTScore F1', 'BLEURT', 'Nubia', 'Nubia (irrelevancy)']]
            )
            # ______________________________________________________ Ref less ______________________________________________________

            abstractness_1 = abstractness_2 = abstractness_3 = '-'
            if 'Data2text' not in dataset and 'Captioning' not in dataset and 'MultiSummEval' not in dataset:
                abstractness_1 = correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['abstractness_1']
                abstractness_2 = correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['abstractness_2']
                abstractness_3 = correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['abstractness_3']
            d_ref0 = {
                **{
                    'Abstr-1': abstractness_1,
                    'Abstr-2': abstractness_2,
                    'Abstr-3': abstractness_3
                },
                **{
                    'Length': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['length'],
                    'Repet-1': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['repetition_1'],
                    'Repet-2': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['repetition_2'],
                    'Repet-3': correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['repetition_3'],
                    '-GPT2 Perpl.': -correl_file[key_nb_ref]['d_scores'][name_dim_dataset]['perplexity'],
                    # 'SUMQA': 'Nan',
                }
            }

            nb_ref_name = '0'
            d_scores_global[name_dim_glob][nb_ref_name][name_col] = d_ref0

            indexes.append(
                [(nb_ref_name, m) for m in ['Abstr-1', 'Abstr-2', 'Abstr-3', 'Length',
                                            'Repet-1', 'Repet-2', 'Repet-3', '-GPT2 Perpl.']]
            )

    return d_scores_global


