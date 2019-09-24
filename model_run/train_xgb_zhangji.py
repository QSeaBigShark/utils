# -*- coding: utf-8 -*-
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


def data_pre(fea_list_path, *datafile):
    datalist = []
    if fea_list_path != "":
        fea_list = [x.strip() for x in open(fea_list_path) if len(x.strip()) > 0]
    for data in datafile:
        data_x, data_Y = data_preprocess(data, fea_list, filter_name=['name', 'idcard', 'phone', 'loan_dt', 'label'])
        datalist.append(data_x)
        datalist.append(data_Y)
    datalist.append(fea_list_path)
    return datalist 
   

def data_preprocess(data, fea_list, filter_name=['name', 'idcard', 'phone', 'loan_dt', 'label']):
    df = pd.read_csv(data, sep='\t')
    df_label = df['label']
    for key in filter_name:
        if key in df.keys():
            del df[key]
    df = df[fea_list]
    df_label = df_label[~df.isnull().all(axis=1)]
    df = df[~df.isnull().all(axis=1)]
    return df, df_label


def train_xgb(sample_data, model_path, params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth':3, 'min_child_weight': 20, 'gamma':0,'subsample': 0.8, 'colsample_bytree':1.0, 'reg_alpha': 0, 'reg_lambda': 1,'silent':1, 'scale_pos_weight':1, 'seed':0}):
    train_x, train_Y, valid_x, valid_Y, f_list = sample_data
    ks_list = []
    auc_list = []

    plst = params.items()
    dTrain = xgb.DMatrix(train_x, label=train_Y)
    dValid = xgb.DMatrix(valid_x, label=valid_Y)
    num_round = params['n_estimators']
    model_clf = xgb.train(plst, dTrain, num_round)
    model_clf.save_model(os.path.join(model_path, "model_clf"))

    f_weights_dict = model_clf.get_score(importance_type='weight')
    f_gains_dict = model_clf.get_score(importance_type='gain')
    f_cover_dict = model_clf.get_score(importance_type='cover')
    #print len(f_weights_dict)
    #print len(f_gains_dict)
    #print len(f_cover_dict)
    fea_analysis = []
    for f_key in f_weights_dict:
        fea_analysis.append({'feature':f_key, 'weight':f_weights_dict[f_key], 'gain':f_gains_dict[f_key], 'cover':f_cover_dict[f_key]})

    fea_analysis_df = pd.DataFrame(fea_analysis, columns=['feature', 'weight', 'gain', 'cover'])
    fea_analysis_df.to_csv(model_path+r'/fip')

    Y_train_prob = model_clf.predict(dTrain)
    Y_valid_prob = model_clf.predict(dValid)

    train_ks = run_ks(train_Y ,Y_train_prob)['ks']
    print "train_ks"
    print train_ks
    valid_ks = run_ks(valid_Y, Y_valid_prob)['ks']
    print "valid_ks"
    print valid_ks
    train_auc = roc_auc_score(train_Y, Y_train_prob)
    valid_auc = roc_auc_score(valid_Y, Y_valid_prob)
    print "train_auc"
    print train_auc
    print "valid_auc"
    print valid_auc
    ks_list.append(train_ks)
    ks_list.append(valid_ks)

    auc_list.append(train_auc)
    auc_list.append(valid_auc)
 

    fw = open(model_path+r'/result', 'a')   
    str_ = str(train_ks) + '\t' + str(valid_ks) + '\t' + str(train_auc) + '\t' + str(valid_auc) + '\t' +json.dumps(params) + '\n'
    print str_
    fw.write(str_)
    print "run success!"


if __name__ == "__main__":
    tmp_params = {"reg_alpha": 0, "colsample_bytree": 1, "silent": 1, 
                  "eval_metric": "auc", "scale_pos_weight": 1, "learning_rate": 0.1, 
                  "nthread": 10, "min_child_weight": 50, "n_estimators": 180, 
                  "subsample": 1, "reg_lambda": 0, "seed": 10, "objective": "binary:logistic", 
                  "tree_method": "auto", "max_depth": 3, "gamma": 0.05}

    print tmp_params
    path = r'/home/users/zhangji01.b/tx/data/innerdata/w/withscore'
    train = path+r'/train'
    valid = path+r'/valid'
    model_res_path = path + r'/model1'
    try:
        fea_list_path = path+r'/inmodel_feature'
    except:
        fea_list_path = ""

    train_x, train_y, valid_x, valid_y, flist = data_pre(fea_list_path, train, valid)
    sample = [train_x, train_y, valid_x, valid_y, flist]
    train_xgb(sample, model_res_path, params=tmp_params )
