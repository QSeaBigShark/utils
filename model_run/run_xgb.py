# -*- coding: utf-8 -*-


import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
# 交叉验证
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
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


def xgb_model(sample_tbl, model_path,
              xgb_params={'nthread': 4, 'n_estimators': 80, 'max_depth': 3,
                          'min_child_weight': 2, 'gamma': 0.1, 'subsample': 0.4, 'learning_rate': 0.06,
                          'colsample_bytree': 0.5, 'scale_pos_weight': 1, 'seed': 100}):
    # os.path.join(model_res_path,  'score'), os.path.join(model_res_path, 'fip')
    plst = xgb_params.items()
    f_score = os.path.join(model_path, "score")
    f_write = os.path.join(model_path, "fip")

    df_x, df_y, df_name = sample_tbl
    x = np.array(df_x)
    y = np.array(df_y)
    # 深度是n，节点数2**(n+1)-1，叶子节点数2**n
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    auc_list = []
    auc_list_train = []
    ks_list = []
    ks_list_train = []
    times = 0
    for train_index, dev_index in kf.split(x, y):
        # print "KFold: %d\nauc\tks" % times
        times += 1
        x_train, x_dev = x[train_index], x[dev_index]
        y_train, y_dev = y[train_index], y[dev_index]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_dev, label=y_dev)

        # evallist = [(dtrain, 'train'), (dvalid, 'eval')]
        #clf_xgb = xgb.train(plst, dtrain, num_boost_round=xgb_params['n_estimators'], evals=evallist)
        clf_xgb = xgb.train(plst, dtrain, num_boost_round=xgb_params['n_estimators'])


        clf_xgb.save_model(os.path.join(model_path, 'clf_'+str(times)))
        clf_xgb.dump_model(os.path.join(model_path, 'dump_raw_'+str(times)))

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
        fea_analysis_df.to_csv(f_write+"_"+str(times), index=None, sep='\t')

        y_proba = clf_xgb.predict(dvalid)
        y_proba_train = clf_xgb.predict(dtrain)

        ks_dict = run_ks(y_dev, y_proba)
        ks_dict_train = run_ks(y_train, y_proba_train)

        auc = roc_auc_score(y_dev, y_proba)
        auc_train = roc_auc_score(y_train, y_proba_train)
        # print "%f\t%f" % (auc, ks)

        ks_list.append(ks_dict['ks'])
        auc_list.append(auc)
        ks_list_train.append(ks_dict_train['ks'])
        auc_list_train.append(auc_train)
        print_ks(ks_dict, f_score)

    fea_analysis_df.to_csv(f_write, index=None, sep='\t')
    clf_xgb.save_model(os.path.join(model_path, "clf"))
    clf_xgb.dump_model(os.path.join(model_path, "dump_raw"))
    dd = dict()
    dd['train_ks'] = ks_list_train
    dd['valida_ks'] = ks_list
    dd['train_auc'] = auc_list_train
    dd['valida_auc'] = auc_list
    train_ks_df = pd.DataFrame(dd)
    train_ks_df.to_csv(os.path.join(model_path, "ks_auc"),index=False, sep='\t')
    
    #params = clf_xgb.get_params()
    #print params
    ks_mean = np.mean(ks_list)
    ks_var = np.std(ks_list)
    auc_mean = np.mean(auc_list)
    auc_var = np.std(auc_list)
    ks_mean_train = np.mean(ks_list_train)
    ks_var_train = np.std(ks_list_train)
    auc_mean_train = np.mean(auc_list_train)
    auc_var_train = np.std(auc_list_train)

    print('train: ')
    print "ks mean: %f, ks var: %f" % (ks_mean_train, ks_var_train)
    print "auc mean: %f, auc var: %f" % (auc_mean_train, auc_var_train)
    print('validation:')
    print "ks mean: %f, ks var: %f" % (ks_mean, ks_var)
    print "auc mean: %f, auc var: %f" % (auc_mean, auc_var)


if __name__ == '__main__':
    # train
    tmp_params = {"reg_alpha": 1, "eval_metric": "auc", "scale_pos_weight": 1, 
                  "learning_rate": 0.09, "n_estimators": 85, "seed": 10, 
                  "tree_method": "auto", "colsample_bytree": 1.0, "silent": 1, 
                  "nthread": 10, "min_child_weight": 50, "subsample": 1, 
                  "reg_lambda": 1, "objective": "binary:logistic", "max_depth": 3, "gamma": 0.15}
    '''
    tmp_params = {'seed': 0,
                  'reg_alpha': 0.1,
                  'n_jobs': 10,
                  'colsample_bytree': 1,
                  'silent': 1,
                  'eval_metric': 'auc',
                  'scale_pos_weight': 1,
                  'random_state': 100,
                  'nthread': 10,
                  'min_child_weight': 100,
                  'n_estimators': 70,
                  'subsample': 0.8,
                  'reg_lambda': 1,
                  'eta': 0.1,
                  'objective': 'binary:logistic',
                  'max_depth':3, 
                  'gamma': 0,
                  'booster': 'gbtree'}
    '''
    print tmp_params
    path = r'/home/users/zhangji01.b/tx/data/innerdata'
    #tbl_name = sys.argv[1]
    tbl_name= path+'/train/train'
    #model_res_path = sys.argv[2]
    model_res_path = path+r'/innermodel/tunparams'
    try:
        #fea_list_path = sys.argv[3]
        fea_list_path = path+r'/inmodel_feature'
    except:
        fea_list_path = ""
    df_x, df_y, f_list, df_median = data_preprocess(tbl_name, fea_list_path, model_res_path)
    tbl_name_list = [df_x, df_y, f_list]
    xgb_model(tbl_name_list, model_res_path, xgb_params=tmp_params)
    with open(os.path.join(model_res_path, "flist"), "w") as f:
        for lines in f_list:
            f.writelines(lines+"\n")
