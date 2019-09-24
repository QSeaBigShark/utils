# -*- coding:utf-8 -*-

#########################################################
# author: tanghuilin
# update_dt: 2017-08-25
#########################################################
# 奇异点
#########################################################

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

def singularity(fea_col, alpha=1.5):
    dic_res = {
            'med': '',
            'seg_5':'',
            'seg_25': '',
            'seg_75': '',
            'seg_95':'',
            'up_limit': '',
            'low_limit': '',
            'up_ratio': '',
            'low_ratio': ''
            }

    class_num = len(fea_col.value_counts())

    fea_null = pd.isnull(fea_col)
    fea_val = fea_col[fea_null == False]
    val_cnt = len(fea_val)

    if val_cnt <= 4:
        return dic_res

    fea_col_sort = fea_col[fea_null == False].sort_values()
    seg_5 = fea_col_sort.iloc[val_cnt / 20]
    seg_25 = fea_col_sort.iloc[val_cnt / 4]
    med = fea_col_sort.iloc[val_cnt / 2]
    seg_75 = fea_col_sort.iloc[val_cnt * 3 / 4]
    seg_95 = fea_col_sort.iloc[val_cnt * 19 / 20]

    if class_num > 2:
        up_limit = seg_75 + (seg_75 - seg_25) * alpha
        low_limit = seg_25 - (seg_75 - seg_25) * alpha
        up_ratio = len(fea_val[fea_val > up_limit]) / float(val_cnt)
        low_ratio = len(fea_val[fea_val < low_limit]) / float(val_cnt)
    else:
        up_limit = ''
        low_limit = ''
        up_ratio = ''
        low_ratio = ''

    dic_res = {
            'med': med,
            'seg_5':seg_5,
            'seg_25': seg_25,
            'seg_75': seg_75,
            'seg_95':seg_95,
            'up_limit': up_limit,
            'low_limit': low_limit,
            'up_ratio': up_ratio,
            'low_ratio': low_ratio
            }

    return dic_res

def drop_singu(X, X_test, alpha=1.5, dir_dst='./'):
    test_switch = bool(len(X_test))
    f_singu = open(dir_dst + 'singularity.txt', 'w')
    for col_name in X.columns:
        dic_res = singularity(X[col_name], alpha)
        up_limit = dic_res['up_limit']
        low_limit = dic_res['low_limit']
        if up_limit == '' and low_limit == '':
            continue
        f_singu.write('%s\t%s\t%s\n' % (col_name, str(up_limit), str(low_limit)))
        X[col_name][(X[col_name] > up_limit) | (X[col_name] < low_limit)] = np.nan
        if test_switch:
            X_test[col_name][(X_test[col_name] > up_limit) | (X_test[col_name] < low_limit)] = np.nan
    f_singu.close()
