# -*- coding: utf-8 -*-


import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import json
import cPickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import cPickle
from ks_ import run_ks, print_ks


def get_statistic(sample_tbl, filter_name=['name', 'id_card', 'phone', 'loan_dt', 'label', 'op_id']):
    df = pd.read_table(sample_tbl, sep='\t')
    # 逾期率
    df_overdue = df['label'][df['label'] == 1]
    print 'Overdue ratio:'
    print 1.0 * df_overdue.shape[0] / df.shape[0]


def data_preprocess(sample_tbl,
                    filter_name=['create_mth', 'create_tm', 'name', 'idcard', 'phone', 'loan_dt', 'label']):
    df = pd.read_table(sample_tbl, sep='\t')
    #df.fillna(-1, inplace=True)
    df_label = df['label']
    for key in filter_name:
        if key in df.keys():
            del df[key]
    # 填充值
    #fill_dict = dict(df.median())
    fill_dict = {}
    return df, df_label, list(df.keys()), fill_dict


def test_xgb(test_tbl, xgb_model, f_list_file):
    df_all = pd.read_table(test_tbl, sep='\t')
    df_all = df_all[['name', 'idcard', 'phone', 'loan_dt', 'label']]
    df_test_x, df_test_y, f_list_test, _ = data_preprocess(test_tbl)
    df_test = pd.DataFrame()
    train_list = []
    with open(f_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                train_list.append(line.strip())
    for e in train_list:
        df_test[e] = df_test_x[e]
    df_test_x = df_test
    df_test_x.fillna(np.nan, inplace=True)
    print 'Read test done'
    test_x = np.array(df_test_x)
    test_y = np.array(df_test_y)
    clf_xgb = xgb.Booster()
    clf_xgb.load_model(xgb_model)
    dtest = xgb.DMatrix(test_x)
    y_proba = clf_xgb.predict(dtest)
    ks_dict = run_ks(test_y, y_proba)
    auc = roc_auc_score(test_y, y_proba)
    print "%f\t%f" % (auc, ks_dict['ks'])
    df_all['prob'] = y_proba
    print_ks(ks_dict, test_tbl + '_score_ks_detail')
    df_all.to_csv(test_tbl+'_prob_', index=None, sep='\t')

def test_xgb_getporb(test_tbl, xgb_model, f_list_file):
    df = pd.read_table(test_tbl, sep='\t')
    df_all = df[['name', 'idcard', 'phone', 'loan_dt']].copy(deep=True)
    df_test_x = pd.DataFrame()
    for k in df.keys():
        if k in ['name', 'idcard', 'phone', 'loan_dt', 'uniq_id']:
            del df[k]
    df_test = pd.DataFrame()
    df_test_x = df
    train_list = []
    with open(f_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                train_list.append(line.strip())
    for e in train_list:
        df_test[e] = df_test_x[e]
    df_test_x = df_test
    df_test_x.fillna(-1, inplace=True)
    print 'Read test done'
    test_x = np.array(df_test_x)
    clf_xgb = xgb.Booster()
    clf_xgb.load_model(xgb_model)
    dtest = xgb.DMatrix(test_x)
    y_proba = clf_xgb.predict(dtest)
    df_all['prob'] = y_proba
    df_all.to_csv(test_tbl+'_prob_', index=None, sep='\t')

if __name__ == '__main__':
    # 调好的参数
    #path=r'/data/zhangji01/tx_feature/data/hunhe/D1'
    path = r'/home/users/zhangji01.b/tx/data/hunhedata/v03/sample-score'
    #train=path+r'/train'
    xgb_file = path+'/model_clf'
    #f_list_file = sys.argv[3]
    train = path + r'/train_select_10per'
    valid = path + r'/valid_select_10per'
    test = path + r'/test_select_10per'
    oot = path + r'/oot5_select_10per'
    oot_xs = path + r'/oot_xs_select_10per'
    f_list_file =path+r'/inmodel_feature'
    #for i in ['07', '08', '09', '10', '11', '12', '01', '02']:
        #t = test_tbl0+'testfq'+i+'_5w'
    #for i in product_name:
    #    t = path+r'/jigou/each_product/'+i
    #    get_statistic(t)
    #    test_xgb(t, xgb_file, f_list_file)
    get_statistic(valid)
    test_xgb(valid, xgb_file, f_list_file)
    #test_xgb_getporb(oot_xs, xgb_file, f_list_file)
    #test_xgb_getporb(train, xgb_file, f_list_file)
