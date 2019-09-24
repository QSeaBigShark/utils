# -*- coding: utf-8 -*-


import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split   #Additional scklearn functions
from sklearn.metrics import roc_auc_score, roc_curve  #Perforing grid search
from ks import run_ks, print_ks


def data_preprocess(sample_tbl, fea_list_path, model_res_path,
                    filter_name=['name', 'idcard', 'phone', 'loan_dt', 'label', 'create_tm']):
    df = pd.read_table(sample_tbl, sep='\t')
    df_label = df['label']

    for key in filter_name:
        if key in df.keys():
            del df[key]

    if fea_list_path != "":
        fea_list = [x.strip() for x in open(fea_list_path) if len(x.strip()) > 0]
        df = df[fea_list]

    df_label = df_label[~df.isnull().all(axis=1)]
    df = df[~df.isnull().all(axis=1)]

    # 填充值
    fill_dict = dict(df.median())

    with open(os.path.join(model_res_path, "fea_fill_lst"), "w") as f:
        for key in fill_dict:
            f.writelines("\t".join(map(str, [key, fill_dict[key]]))+"\n")

    return df, df_label, list(df.keys()), fill_dict

def select_model(train_x, train_y, test_x, test_y):
    cv_params = {'n_estimators': [50,100,200,300,400]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth':3, 'min_child_weight': 20, 'gamma':0,
                    'subsample': 0.8, 'colsample_bytree':1.0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'silent':1, 'scale_pos_weight':1, 'seed':0}

    model = xgb.XGBClassifier(**other_params)
    optimized_XGB = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_XGB.fit(train_x, train_y)
    #evalute_result = optimized_XGB.cv_results_
    #print "每轮迭代运行结果:{0}".format(evalute_result)
    print "参数的最佳取值:{0}".format(optimized_XGB.best_params_)
    print "最佳模型得分:{0}".format(optimized_XGB.best_score_)

    predict_train_y = optimized_XGB.predict(train_x)
    train_ks = run_ks(train_y, predict_train_y)

    predict_test_y = optimized_XGB.predict(test_x)
    test_ks = run_ks(test_y, predict_test_y)

    #print "训练集ks: %f，测试集ks: %f" % (train_ks, test_ks)
    print train_ks
if __name__ == "__main__":

    # 路径
    path = r'/home/users/zhangji01.b/tx/data/innerdata'

    # 训练集
    train_sample = path + r'/train/train'
    #测试集
    test_sample = path + r'/test/test'
    # 模型文件保存路径
    model_res_path = path + r'/innermodel/model/'
    try:
        # 入模特征列表
        fea_list_path = path + r'/inmodel_feature'
    except:
        fea_list_path = ""
    # train_sample训练集, fea_list_path入模特征，model_res_path模型输出路径
    train_x, train_y, f_list, df_median = data_preprocess(train_sample, fea_list_path, model_res_path)
    test_x, test_y, _, _, = data_preprocess(test_sample, fea_list_path, model_res_path)

    select_model(train_x, train_y, test_x, test_y)
