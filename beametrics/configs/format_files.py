import pandas as pd
import numpy as np
import json
import ast
import pickle
import copy

#import krippendorff





def format_E2E(path, keep_only, dimensions):
    with open(path, "r") as f_r:
        temp_data = json.loads(f_r.read())

    d_data = {}
    for i, ex in enumerate(temp_data):
        for d in dimensions:
            ex[d] = np.average(ex[d])
        d_data[i] = ex
    return  d_data


def format_wikibio(path, keep_only, dimensions):
    with open(path, "r") as f_r:
        temp_data = json.loads(f_r.read())

    d_data = {}
    for i, ex in enumerate(temp_data):
        for d in dimensions:
            ex[d] = float(ex[d])
        d_data[i] = ex
    return  d_data




def fill_rank_captioning_pascal50s(d_data, dim_1s, dim_2s):

    for ex_id in range(0, len(d_data), 2):
        for metric in set(dim_1s + dim_2s):
            m1 = d_data[str(ex_id)][metric]
            m2 = d_data[str(ex_id+1)][metric]

            if m1 >= m2 :
                rank1 = 1
                rank2 = 0
            else:
                rank1 = 0
                rank2 = 1

            d_data[str(ex_id)][metric] = rank1
            d_data[str(ex_id+1)][metric] = rank2

    return d_data