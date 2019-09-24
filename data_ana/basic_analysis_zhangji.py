# -*- coding:utf-8 -*-

import sys, os
import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn import metrics
from ks import *
from singularity import *

SINGU_ALPHA = 1.5    # 奇异点系数


class BasicAnalysis(object):
    def __init__(self, data, label_col_name, invaild_feature):
        self.data = data
        for feature_name in invaild_feature:
            if feature_name in self.data.columns:
                self.data.drop([feature_name], inplace=True, axis=1)

        self.label_col_name = label_col_name
        self.sample_num = len(self.data)

    def _overdue_ratio(self, label_col):
        if len(label_col) > 0:
            return sum(label_col) / float(len(label_col))
        else:
            return -1

    def _sin_fea_sta(self, fea_col, fea_null):
        val_cnt = len(fea_col[fea_null == False])
        zero_cnt = len(fea_col[fea_col == 0])

        coverage_ratio = val_cnt / float(self.sample_num)
        if val_cnt > 0:
            not_zero_ratio = (val_cnt - zero_cnt) / float(val_cnt)
        else:
            not_zero_ratio = -1

        class_num = len(fea_col.value_counts())

        avg_val = fea_col.mean()
        min_val = fea_col.min()
        max_val = fea_col.max()

        dic_odd = singularity(fea_col, alpha=SINGU_ALPHA)

        return [
            coverage_ratio, not_zero_ratio, class_num,
            avg_val, min_val, dic_odd['seg_5'],dic_odd['seg_25'],
            dic_odd['med'], dic_odd['seg_75'], dic_odd['seg_95'], max_val
        ]

    def sta_2v(self, label, fea):
        [[tn, fp], [fn, tp]] = metrics.confusion_matrix(label, fea)
        if tp + fn:
            tpr = tp / float(tp + fn)
        else:
            tpr = -1
        if fp + tn:
            fpr = fp / float(fp + tn)
        else:
            fpr = -1
        if tp + fp:
            precision = tp / float(tp + fp)
        else:
            precision = -1
        if tn + fn:
            ovd_ratio_0 = fn / float(tn + fn)
        else:
            ovd_ratio_0 = -1
        if fp + tp:
            ovd_ratio_1 = tp / float(tp + fp)
        else:
            ovd_ratio_1 = -1
        accuracy = (tp + tn) / float(tp + fn + fp + tn)
        chi2, pval = feature_selection.chi2(fea.values.reshape(-1, 1), label)
        dic_2v = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'tpr': tpr,
                'fpr': fpr,
                'ovd_ratio_0': ovd_ratio_0,
                'ovd_ratio_1': ovd_ratio_1,
                'precision': precision,
                'accuracy': accuracy,
                'chi2': chi2[0],
                'pval': pval[0]
                }
        return dic_2v

    def ttl_overdue_ratio(self):
        return self._overdue_ratio(self.data[self.label_col_name])

    def sin_fea_analysis(self):
        sin_fea_ana_dic = {}
        for col_name in self.data.columns:
            if col_name == self.label_col_name:
                continue
            fea_null = pd.isnull(self.data[col_name])
            sin_fea_sta = self._sin_fea_sta(self.data[col_name], fea_null)

            label_hit = self.data[self.label_col_name][fea_null == False]
            if len(label_hit) == 0:
                print 'hit_0:', col_name
                continue
            label_miss = self.data[self.label_col_name][fea_null == True]
            sin_overdue_ratio = [self._overdue_ratio(label_hit), self._overdue_ratio(label_miss)]
            miss_order_num = len(label_miss)
            miss_ovd_num = sum(label_miss)
            dic_ks = {}
            dic_2v = {}
            class_num = sin_fea_sta[2]
            if miss_order_num > 0:
                dic_ks = ks(label_hit, self.data[col_name][fea_null == False], miss_order_num, miss_ovd_num)
            else:
                dic_ks = ks2(label_hit, self.data[col_name][fea_null == False])
            if class_num > 2:     # class_num
                sin_fea_sta.insert(0, dic_ks['iv'])
                sin_fea_sta.insert(0, dic_ks['ks'])

            elif class_num == 2:
                try:
                    dic_2v = self.sta_2v(label_hit, self.data[col_name][fea_null == False])
                    sin_fea_sta.insert(0, dic_ks['iv'])
                    sin_fea_sta.insert(0, -1)

                except:
                    sin_fea_sta.insert(0, -1)
                    sin_fea_sta.insert(0, -1)

            else:
                print 'class_num_1:', col_name
                continue

            sin_fea_ana_dic[col_name] = [sin_fea_sta + sin_overdue_ratio, dic_ks, dic_2v]
        return sin_fea_ana_dic

def basic_main(fea_fin, out_dir, invaild_feature, sep=','):
    """
    主函数
    :param fea_fin: 待计算文件名
    :param out_dir: 输出路径
    :param invaild_feature: 不计算特征名
    :param sep: 输入文件分隔符
    :return:
    """
    item_list = ['feature_name', 'ks', 'iv', 'coverage', 'not_zero_ratio', 'class_num', 'avg_val', 'min_val', 'seg_5', 'seg_25',
                 'med', 'seg_75', 'seg_95', 'max_val', 'hit_overdue_ratio', 'miss_overdue_ratio']
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    f = open(out_dir + 'sin_fea_ana.txt', 'w')

    f.write('\t'.join(item_list) + '\n')

    f_ks = open(out_dir + 'sin_fea_ks.txt', 'w')  # ks detail file
    f_2v = open(out_dir + 'sin_fea_2v.txt', 'w')  # confusion matrix and chi2 file
    f_2v.write('\t'.join(
        ['feature', 'tp', 'fp', 'tn', 'fn', 'tpr(recall)', 'fpr', 'ovd_ratio_0', 'ovd_ratio_1', 'precision', 'accuracy',
         'chi2', 'pval']) + '\n')
    #TODO*******************************
    df_fea = pd.read_csv(fea_fin, index_col=0, sep=sep)
    print 'read file: ' + fea_fin
    ba = BasicAnalysis(df_fea, 'label', invaild_feature)  # feature file name, label column name

    sfa_dic = ba.sin_fea_analysis()  # single feature analysis

    for fea in sfa_dic:

        f.write('%s\t%s\n' % (str(fea), '\t'.join([str(x) for x in sfa_dic[fea][0]])))

        if sfa_dic[fea][0][4] > 2:  # write ks detail file
            ks_info = sfa_dic[fea][1]
            f_ks.write(fea + '\nks: %.2f%%\niv: %.4f\t\t\t\t\t\t\t\t\t\t\t\t\n' % (ks_info['ks'], ks_info['iv']))
            print fea
            print 'ks:' + str(ks_info['ks'])
            print 'iv ' + str(ks_info['iv'])
            f_ks.write('\t'.join(
                ['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比(%)', '逾期率(%)', '累计坏账户占比(%)', '累计好账户占比(%)', 'KS统计量(%)', 'WOE',
                 'IV统计量']) + '\n')
            for i in range(len(ks_info['ks_list'])):
                f_ks.write('%.2f\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\n' % (
                i + 1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i],
                ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i],
                ks_info['good_ratio'][i], ks_info['ks_list'][i], ks_info['woe'][i], ks_info['iv_list'][i]))
        elif sfa_dic[fea][0][4] == 2:  # write confusion matrix and chi2 file
            sta_2v = sfa_dic[fea][2]
            try:
                f_2v.write('%s\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%e\n' % (
                fea, sta_2v['tp'], sta_2v['fp'], sta_2v['tn'], sta_2v['fn'], sta_2v['tpr'], sta_2v['fpr'],
                sta_2v['ovd_ratio_0'], sta_2v['ovd_ratio_1'], sta_2v['precision'], sta_2v['accuracy'], sta_2v['chi2'],
                sta_2v['pval']))
            except:
                print fea, '2v error'
        else:
            print 'class_num == 0:', fea
    f_ks.close()
    f_2v.close()
    f.close()


if __name__ == '__main__':
    #fea_fin = sys.argv[1]
    #out_dir = sys.argv[2]   # 输出文件目录，带/
    invalid_feature = ['name', 'idcard', 'phone', 'loan_dt', 'uniq_id']
   
    pathaa = r'/home/users/zhangji01.b/tx/data/hunhedata/v02/sample'
    fea_fin = pathaa + r'/train_tsn_score'
    out_dir = pathaa + r'/ana/'
    sep = '\t'
    print 'begin'
    basic_main(fea_fin, out_dir, invalid_feature, sep)
    print 'over'
