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
from ks import run_ks, print_ks


def get_statistic(sample_tbl, filter_name=['name', 'id_card', 'phone', 'loan_dt', 'label', 'op_id']):
    df = pd.read_table(sample_tbl, sep='\t')
    # 逾期率
    df_label = df['label']
    df_overdue = df['label'][df['label'] == 1]
    print 'Overdue ratio:'
    print 1.0 * df_overdue.shape[0] / df.shape[0]
    for key in filter_name:
        if key in df.keys():
            del df[key]


def data_preprocess(sample_tbl,
                    filter_name=['create_mth', 'create_tm', 'name', 'idcard', 'phone', 'loan_dt', 'label']):
    df = pd.read_table(sample_tbl, sep='\t')
    # df.fillna(-1, inplace=True)
    df_label = df['label']
    for key in filter_name:
        if key in df.keys():
            del df[key]
    # 填充值
    fill_dict = dict(df.median())
    return df, df_label, list(df.keys()), fill_dict


def train_xgb(sample_tbl, fip_file, clf_model,
              xgb_params={'nthread': 4, 'n_estimators': 80, 'max_depth': 3,
                          'min_child_weight': 2, 'gamma': 0.1, 'subsample': 0.4, 'learning_rate': 0.06,
                          'colsample_bytree': 0.5, 'scale_pos_weight': 1, 'seed': 100}):
    plst = xgb_params.items()
    df_x, df_y, f_list, df_median = data_preprocess(sample_tbl)
    x = np.array(df_x)
    y = np.array(df_y)
    dtrain = xgb.DMatrix(x, label=y)
    evallist = [(dtrain, 'eval')]
    # num_boost_round就是树的棵树
    clf_xgb = xgb.train(plst, dtrain, num_boost_round=xgb_params.get('n_estimators'), evals=evallist)
    print 'xgb done'
    f_weights_dict = clf_xgb.get_score(importance_type='weight')
    f_gains_dict = clf_xgb.get_score(importance_type='gain')
    f_covers_dict = clf_xgb.get_score(importance_type='cover')
    fea_analysis = []
    for f_key in f_weights_dict:
        fea_analysis.append(
            {'feature': f_list[int(f_key[1:])], 'weight': f_weights_dict[f_key], 'gain': f_gains_dict[f_key],
             'cover': f_covers_dict[f_key]})
    fea_analysis_df = pd.DataFrame(fea_analysis, columns=['feature', 'weight', 'gain', 'cover'])
    fea_analysis_df.sort_values(['gain'], ascending=False, inplace=True)
    fea_analysis_df.to_csv(fip_file, index=None, sep='\t')
    clf_xgb.save_model(clf_model + 'clf')
    clf_xgb.dump_model(clf_model + 'dump_raw')
    return clf_xgb, f_list


def test_xgb(test_tbl, xgb_model, train_list):
    df_test_x, df_test_y, f_list_test, df_median = data_preprocess(test_tbl)
    df_test = pd.DataFrame()
    for e in train_list:
        df_test[e] = df_test_x[e]
    df_test_x = df_test
    # df_test_x.fillna(-1, inplace=True)
    print 'Read test done'
    test_y = np.array(df_test_y)
    xgb = xgb_model
    test_x = np.array(df_test_x)
    y_proba = xgb.predict_proba(test_x)
    y_score = [item[0] for item in y_proba]
    y_good = [1 - item for item in test_y]
    tmp_df = pd.DataFrame()
    tmp_df['f'] = y_score
    tmp_df['good'] = y_good
    tmp_df['bad'] = test_y
    ks_dict = run_ks(test_y, y_proba[:, 1])
    auc = roc_auc_score(test_y, y_proba[:, 1])
    print "%f\t%f" % (auc, ks_dict['ks'])
    print_ks(ks_dict, test_tbl + '_score_ks_detail')


if __name__ == '__main__':
    # train
    #tmp_params = {'seed':6, 'reg_alpha': 0, 'n_jobs': 10, 'colsample_bytree': 0.2, 
    #	'silent': True, 'eval_metric': 'auc', 'scale_pos_weight': 1, 'random_state': 100,
    #	'nthread': 10, 'min_child_weight': 30, 'n_estimators': 200, 'subsample': 0.2, 
    #	'reg_lambda': 0, 'eta': 0.15, 'objective': 'binary:logistic', 'max_depth': 2,
    #	'gamma': 0.8, 'booster': 'gbtree'}
    tmp_params = {"reg_alpha": 0, "eval_metric": "auc", "scale_pos_weight": 1, 
        "learning_rate": 0.09, "n_estimators": 47, "seed": 0, "tree_method": "auto", "colsample_bytree": 0.9, 
        "silent": 1, "nthread": 10, "min_child_weight": 100, "subsample": 0.8, 
        "reg_lambda": 0, "objective": "binary:logistic", "max_depth": 3, "gamma": 0}

    print tmp_params
    #tbl_name = sys.argv[1]
    pathaa = r'/data/zhangji01/tx_feature/data/hunhe'
    tbl_name = pathaa+r'/D1/train'
    get_statistic(tbl_name)
    xgb, f_list = train_xgb(tbl_name, tbl_name + '_fip', tbl_name + '_model', xgb_params=tmp_params)
    f = open(tbl_name+'flist', 'w')
    for e in f_list:
        f.write(e+'\n')
    f.close()
    #test_tbl = sys.argv[2]
    #test_tbl = pathaa+r'/test/test'
    #get_statistic(test_tbl)
    #test_xgb(test_tbl, xgb, f_list)
