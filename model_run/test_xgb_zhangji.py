import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
from ks import run_ks, print_ks
from sklearn.metrics import roc_auc_score, roc_curve


def get_statistic(sample_tbl, filter_name=['name', 'id_card', 'phone', 'loan_dt', 'label', 'op_id']):
    df = pd.read_table(sample_tbl, sep='\t')
    df_overdue = df['label'][df['label'] == 1]
    print 'Overdue ratio:'
    print 1.0 * df_overdue.shape[0] / df.shape[0]

def test_xgb_noin(data, fea_list_path, in_fea,  model_path, filter_name=['name', 'idcard', 'phone', 'loan_dt', 'label']):
    fea_list = []
    in_fea_list = []
    df = pd.read_csv(data, sep='\t')
    df_label = df['label']
    df_out = df
    for key in filter_name:
        if key in df.keys():
            del df[key]
    if fea_list_path != "":
        fea_list = [x.strip() for x in open(fea_list_path) if len(x.strip()) > 0]
    print len(fea_list)
    if in_fea != "":
        in_fea_list = [x.strip() for x in open(in_fea) if len(x.strip()) > 0]
    print len(in_fea_list)
    ss = list(set(fea_list)-set(in_fea_list))
    #print ss
    print len(ss)
    df = df[fea_list]
    for i in ss:
        df[i] = 0
    df_label = df_label[~df.isnull().all(axis=1)]
    df_out = df_out[~df.isnull().all(axis=1)]
    df = df[~df.isnull().all(axis=1)]
    test_x = df
    test_Y = df_label
    f_list = fea_list
    
    dTest = xgb.DMatrix(test_x, label=test_Y)
    clf_xgb = xgb.Booster()
    clf_xgb.load_model(model_path)
    print "Read test done"
    dTest = xgb.DMatrix(test_x)
    y_proba = clf_xgb.predict(dTest)
    ks_ = run_ks(test_Y, y_proba)
    #auc_ = roc_auc_score(test_Y, y_proba)
    print "%f" % ( ks_['ks'])
    #df_out['prob'] = y_proba
    #print_ks(ks_, data+r'_score_ks_detail')
    #df_out.to_csv(data+r'_prob_', index=False, sep='\t')


def test_xgb(data, fea_list_path, model_path, filter_name=['name', 'idcard', 'phone', 'loan_dt', 'label']):
    fea_list = []
    df = pd.read_csv(data, sep='\t')
    df_label = df['label']
    df_out = df
    for key in filter_name:
        if key in df.keys():
            del df[key]
    if fea_list_path != "":
        fea_list = [x.strip() for x in open(fea_list_path) if len(x.strip()) > 0]
    df = df[fea_list]
    df_label = df_label[~df.isnull().all(axis=1)]
    df_out = df_out[~df.isnull().all(axis=1)]
    df = df[~df.isnull().all(axis=1)]
    test_x = df
    test_Y = df_label
    f_list = fea_list

    dTest = xgb.DMatrix(test_x, label=test_Y)
    clf_xgb = xgb.Booster()
    clf_xgb.load_model(model_path)
    print "Read test done"
    dTest = xgb.DMatrix(test_x)
    y_proba = clf_xgb.predict(dTest)
    print y_proba
    ks_ = run_ks(test_Y, y_proba)
    auc_ = roc_auc_score(test_Y, y_proba)
    print "%f\t%f" % (auc_, ks_['ks'])
    df_out['prob'] = y_proba
    print_ks(ks_, data+r'_score_ks_detail')
    df_out.to_csv(data+r'_prob_', index=False, sep='\t')


def test_xgb_prob(data, fea_list_path,  model_path, filter_name=['name', 'idcard', 'phone', 'loan_dt', 'uniq_id']):
    fea_list = []
    df = pd.read_csv(data, sep='\t')
    df_out = df
    for key in filter_name:
        if key in df.keys():
            del df[key]
    if fea_list_path != "":
        fea_list = [x.strip() for x in open(fea_list_path) if len(x.strip()) > 0]
    df = df[fea_list]
    df_out = df_out[~df.isnull().all(axis=1)]
    df = df[~df.isnull().all(axis=1)]
    #test_x = df
    f_list = fea_list
    test_x = np.array(df)
    dtest = xgb.DMatrix(test_x)
    clf_xgb = xgb.Booster()
    clf_xgb.load_model(model_path)
    print "Read test done"
    y_proba = clf_xgb.predict(dtest)
    df_out['prob'] = y_proba
    df_out.to_csv(data+r'_prob_', index=False, sep='\t')


if __name__ == "__main__":
    path = r'/home/users/zhangji01.b/tx/data/innerdata/w/withscore'
    xgb_file=path+'/model/model_clf'
    se = r''
    #f_list_file = sys.argv[3]
    train = path+r'/train'+se
    valid = path+r'/valid'+se
    test = path+r'/test'+se
    test_ = path+r'/test_'+se
    oot6 = path+r'/oot_month6'
    xs = path+r'/z/xianshang/xs'+se
    f_list_file =path + r'/inmodel_feature'

    in_model = path + r'/model1/inmodel' 
    #get_statistic(test)
    test_xgb(test, f_list_file, xgb_file)
    test_xgb(test_, f_list_file, xgb_file)
    
    '''
    get_statistic(valid)
    test_xgb(valid, f_list_file, xgb_file, f_list_file)
    get_statistic(test)
    test_xgb(test, f_list_file, xgb_file, f_list_file)   
    get_statistic(oot1)
    test_xgb(oot1, f_list_file, xgb_file, f_list_file)
    get_statistic(oot2)
    test_xgb(oot2, f_list_file, xgb_file, f_list_file)
    get_statistic(oot3)
    test_xgb(oot3, f_list_file, xgb_file, f_list_file)
    get_statistic(oot4)
    test_xgb(oot4, f_list_file, xgb_file, f_list_file)
    '''
    #get_statistic(m01)
    ##test_xgb(m01, f_list_file, xgb_file, f_list_file)
    #get_statistic(m02)
    #test_xgb(m02, f_list_file, xgb_file, f_list_file)
    #get_statistic(m1907)
    #test_xgb_noks(m1907, f_list_file, xgb_file, f_list_file)
    #test_xgb_prob(xs, f_list_file, xgb_file, f_list_file)
