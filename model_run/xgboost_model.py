# -*-coding: utf-8 -*-

import logging
import pandas as pd
import xgboost as xgb
#import codecs
import sys, os
#from multiprocessing import cpu_count
import operator
import yaml
from ks import ks, print_ks
from sklearn import metrics

#reload(sys)
#sys.setdefaultencoding('utf8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')

def read_csv(data_path, useless, label_col, model_path):
    data = pd.read_csv(data_path, sep='\t')
    dropped = []
    for item in useless:
        if item in data.columns:
            data.drop([item], inplace=True, axis=1)
            dropped.append(item)
    #logging.info('droped columns : {0}'.format(dropped))
    features = list(data.columns.values)
    features.remove(label_col)
    #logging.info('data has {0} features'.format(len(features)))
    y = data[label_col]
    X = data[features]
    header = data.columns.values.tolist()
    header.remove(label_col)
    fp = open(model_path+'/fmap','w')
    index = 0
    for field in header:
        fp.write(str(index))
        fp.write('\t')
        fp.write(str(field))
        fp.write('\t')
        fp.write('q')
        fp.write('\n')
        index = index + 1
    fp.close()
    return X, y

def extract_y_from_libsvm_file(data_path):
    fp = open(data_path, 'r')
    y_list = []
    for line in fp:
        label_value = int(line.split()[0])
        y_list.append(label_value)
    fp.close()
    return y_list

def xgboost_model(train_data_path, eval_data_path, test_data_path, oot1_data_path, oot2_data_path, data_type, model_output_path, dump_tree_path, num_round, model_path, output, param):
    if data_type == 'csv':
        logging.info('reading train file: {0}'.format(train_data))
        X, y = read_csv(train_data_path, useless, label_col, model_path)
        logging.info('reading eval file: {0}'.format(eval_data))
        X_eval, y_eval = read_csv(eval_data_path, useless, label_col, model_path)
        dtrain = xgb.DMatrix(X, label=y)
        deval = xgb.DMatrix(X_eval, label=y_eval)

    elif data_type == 'libsvm':
        y = extract_y_from_libsvm_file(train_data_path)
        y_eval = extract_y_from_libsvm_file(eval_data_path)
        dtrain = xgb.DMatrix(train_data_path)
        deval = xgb.DMatrix(eval_data_path)
#    evallist = [(deval, 'eval'), (dtrain, 'train')]
    evallist = [(dtrain, 'train'),(deval, 'eval')]
    #print(deval)
    #print(X_eval)
    #print(y_eval)
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=30)
    bst.save_model(model_output_path)
    # 转存模型
    bst.dump_model(dump_tree_path)
    # 生成特征重要性文件
    if data_type == 'csv':
        if output == 'importance' or output == 'all':
            importance = bst.get_fscore(model_path + '/fmap')
            print(len(importance))
            importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
            df = pd.DataFrame(importance, columns=['feature','importance'])
            df.to_csv(model_path+'/feature_importance')
    # 生成KS
    if output == 'ks' or output == 'auc' or output == 'all':
        y_predict = bst.predict(dtrain)
        y_eval_predict = bst.predict(deval)
        if output == 'ks' or output == 'all':
            ks_train = ks(y, y_predict)
            ks_eval = ks(y_eval, y_eval_predict)
            print('------------train ks-------------')
            #print_ks(ks_train)
            print(round(ks_train['ks'], 5))
            print('------------eval  ks-------------')
            #print_ks(ks_eval)
            print(round(ks_eval['ks'], 5))
            print('------------test  ks-------------')
            if test_data_path != '':
                if data_type == 'libsvm':
                    y_test = extract_y_from_libsvm_file(test_data_path)
                    dtest = xgb.DMatrix(test_data_path)
                elif data_type == 'csv':
                    X_test, y_test = read_csv(test_data_path, useless, label_col, model_path)
                    dtest = xgb.DMatrix(X_test, label=y_test)
                y_test_predict = bst.predict(dtest)
                ks_test = ks(y_test, y_test_predict)
                print(round(ks_test['ks'], 5))
            else:
                print('Ignored because of no valid test_data_path provided.')
            print('------------oot1  ks-------------')
            if oot1_data_path != '':
                if data_type == 'libsvm':
                    y_oot1 = extract_y_from_libsvm_file(oot1_data_path)
                    doot1 = xgb.DMatrix(oot1_data_path)
                elif data_type == 'csv':
                    X_oot1, y_oot1 = read_csv(oot1_data_path, useless, label_col, model_path)
                    doot1 = xgb.DMatrix(X_oot1, label=y_oot1)
                y_oot1_predict = bst.predict(doot1)
                ks_oot1 = ks(y_oot1, y_oot1_predict)
                print(round(ks_oot1['ks'], 5))
            else:
                print('Ignored because of no valid oot1_data_path provided.')
            print('------------oot2  ks-------------')
            if oot2_data_path != '':
                if data_type == 'libsvm':
                    y_oot2 = extract_y_from_libsvm_file(oot2_data_path)
                    doot2 = xgb.DMatrix(oot2_data_path)
                elif data_type == 'csv':
                    X_oot2, y_oot2 = read_csv(oot2_data_path, useless, label_col, model_path)
                    doot2 = xgb.DMatrix(X_oot2, label=y_oot2)
                y_oot2_predict = bst.predict(doot2)
                ks_oot2 = ks(y_oot2, y_oot2_predict)
                print(round(ks_oot2['ks'], 5))
            else:
                print('Ignored because of no valid oot2_data_path provided.')
        if output == 'auc' or output == 'all':
            auc_train = metrics.roc_auc_score(y, y_predict)
            auc_eval = metrics.roc_auc_score(y_eval, y_eval_predict)
            print('-------------train auc-------------')
            print(round(auc_train, 4))
            print('-------------eval  auc-------------')
            print(round(auc_eval, 4))
            print('-------------test  auc-------------')
            if test_data_path != '':
                if data_type == 'libsvm':
                    y_test = extract_y_from_libsvm_file(test_data_path)
                    dtest = xgb.DMatrix(test_data_path)

                elif data_type == 'csv':
                    X_test, y_test = read_csv(test_data_path, useless, label_col, model_path)
                    dtest = xgb.DMatrix(X_test, label=y_test)
                y_test_predict = bst.predict(dtest)
                auc_test = metrics.roc_auc_score(y_test, y_test_predict)
                print(round(auc_test, 4))
            else:
                print('Ignored because of no valid test_data_path provided.')
            print('-------------oot1  auc-------------')
            if oot1_data_path != '':
                if data_type == 'libsvm':
                    y_oot1 = extract_y_from_libsvm_file(oot1_data_path)
                    doot1 = xgb.DMatrix(oot1_data_path)

                elif data_type == 'csv':
                    X_oot1, y_oot1 = read_csv(oot1_data_path, useless, label_col, model_path)
                    doot1 = xgb.DMatrix(X_oot1, label=y_oot1)
                y_oot1_predict = bst.predict(doot1)
                auc_oot1 = metrics.roc_auc_score(y_oot1, y_oot1_predict)
                print(round(auc_oot1, 4))
            else:
                print('Ignored because of no valid oot1_data_path provided.')
            print('-------------oot2  auc-------------')
            if oot2_data_path != '':
                if data_type == 'libsvm':
                    y_oot2 = extract_y_from_libsvm_file(oot2_data_path)
                    doot2 = xgb.DMatrix(oot2_data_path)

                elif data_type == 'csv':
                    X_oot2, y_oot2 = read_csv(oot2_data_path, useless, label_col, model_path)
                    doot2 = xgb.DMatrix(X_oot2, label=y_oot2)
                y_oot2_predict = bst.predict(doot2)
                auc_oot2 = metrics.roc_auc_score(y_oot2, y_oot2_predict)
                print(round(auc_oot2, 4))
            else:
                print('Ignored because of no valid oot2_data_path provided.')
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        logging.info(path + ' 创建成功')
        return True
    else:
        logging.info(path + ' 目录已存在')
        return False

if __name__ == '__main__':

    # Get all parameters from config file.
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath,'params.yaml')
    f = open(yamlPath,'r')
    cont = f.read()
    cf = yaml.load(cont)

    data_path    = cf['data']['data_path']
    data_type    = cf['data']['data_type']
    train_data   = cf['data']['train_data']
    eval_data    = cf['data']['eval_data']
    test_data    = cf['data']['test_data']
    oot_data1     = cf['data']['oot_data1']
    oot_data2     = cf['data']['oot_data2']
    useless      = cf['data']['useless']
    label_col    = cf['data']['label_col']
    model_name   = cf['model']['model_name']
    model_path   = cf['model']['model_path']
    num_round    = cf['model']['num_round']

    param        = {}
    param['max_depth']    = cf['model']['param']['max_depth']
    param['eta']          = cf['model']['param']['eta']
    param['objective']    = cf['model']['param']['objective']
    param['lambda']       = cf['model']['param']['lambda']
    param['min_child_weight'] \
                              = cf['model']['param']['min_child_weight']
    param['subsample']    = cf['model']['param']['subsample']
    param['nthread']      = cf['model']['param']['nthread']
    param['eval_metric']  = cf['model']['param']['eval_metric']
    #param['gpu_id']  = cf['model']['param']['gpu_id']
    #param['tree_method']  = cf['model']['param']['tree_method']
    #param['max_bin']  = cf['model']['param']['max_bin']
    #param['n_gpus']  = cf['model']['param']['n_gpus']

    output       = cf['output']

    mkdir(model_path)
    if data_type == 'csv' or data_type == 'libsvm':
        pass
    else:
        logging.error('Error: data_type can only be set as \'csv\' or \'libsvm\'.')
        exit(-1)

    train_data_path = os.path.join(data_path, train_data)
    eval_data_path = os.path.join(data_path, eval_data)
    test_data_path = os.path.join(data_path, test_data)
    oot1_data_path = os.path.join(data_path, oot_data1)
    oot2_data_path = os.path.join(data_path, oot_data2)

    model_name = 'model.' + model_name + '.trees{0}.depth{1}'.format(num_round,param['max_depth'])
    dump_tree_name = 'dump.' + model_name + '.trees{0}.depth{1}'.format(num_round,param['max_depth'])
    model_output_path = os.path.join(model_path, model_name)
    dump_tree_path = os.path.join(model_path,dump_tree_name)

    if not os.path.exists(train_data_path) or not os.path.exists(eval_data_path):
        logging.error('Error: Please provide valid train_data_path and eval_data_path.')
        exit(-1)

    if not os.path.exists(test_data_path):
        test_data_path = ''
    
    if not os.path.exists(oot1_data_path):
        oot1_data_path = ''

    if not os.path.exists(oot2_data_path):
        oot2_data_path = ''

    logging.info('training model: {0}'.format(model_name))
    xgboost_model(train_data_path, eval_data_path, test_data_path, oot1_data_path, oot2_data_path, data_type, model_output_path, dump_tree_path, num_round, model_path, output, param)
