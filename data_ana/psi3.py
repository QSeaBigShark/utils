# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import math

PART = 10

def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = (head_pos + tail_pos) / 2
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)

#part_num分成几组
def psi(score_1, score_2, part_num=PART):
    null_1 = pd.isnull(score_1)
    score_1 = score_1[null_1 == False] #fea1不为空的留下
    score_1 = np.array(score_1.T) #转置
    score_1.sort()  #fea1排序
    null_2 = pd.isnull(score_2)
    score_2 = score_2[null_2 == False]#score_2不为空的留下
    score_2 = np.array(score_2.T)
    # score_1
    length = len(score_1)
    length2 = len(score_2)
    if length2*length ==0:
        if length2 ==length:
            return 0
        else :
            return 10086
    cut_list = [min(score_1)]
    order_num = []
    #print "-------"
    #print order_num
    cut_pos_last = -1
    for i in np.arange(part_num):
        if i == part_num-1 or score_1[length*(i+1)/part_num-1] != score_1[length*(i+2)/part_num-1]:
            cut_list.append(score_1[length*(i+1)/part_num-1])
            if i != part_num-1:
                cut_pos = _get_cut_pos(score_1[length*(i+1)/part_num-1], score_1, length*(i+1)/part_num-1, length*(i+2)/part_num-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            cut_pos_last = cut_pos
    order_num = np.array(order_num)
    #print order_num
    order_ratio_1 = order_num / float(length)
    #print 'cut_list', cut_list
    #print 'order_ratio_1', order_ratio_1, sum(order_ratio_1)

    # score_2
    length = len(score_2)
    order_num = []
    for i in range(len(cut_list)):
        if i == 0:
            continue
        elif i == 1:
            order_num.append(len(score_2[(score_2 <= cut_list[i])]))
        elif i == len(cut_list)-1:
            order_num.append(len(score_2[(score_2 > cut_list[i-1])]))
        else:
            order_num.append(len(score_2[(score_2 > cut_list[i-1]) & (score_2 <= cut_list[i])]))
    order_num = np.array(order_num)
    #print order_num
    order_ratio_2 = order_num / float(length)
    #print 'order_ratio_2', order_ratio_2, sum(order_ratio_2)

    # psi
    try:
        psi = sum([(order_ratio_1[i] - order_ratio_2[i]) * math.log((order_ratio_1[i] / order_ratio_2[i]), math.e) for i in range(len(order_ratio_1))])
    except:
        psi = -999.0
    return psi

if __name__ == '__main__':
    pathaa = r'/home/users/zhangji01.b/tx/data/hunhedata/v03/sample-score'
    #pathaa = r'/home/users/zhangji01.b/tx/data/innerdata/month'
    df_1 = pd.read_csv(pathaa+'/innertrain_select_10per', sep='\t')
    df_2 = pd.read_csv(pathaa+'/oot5_select_10per', sep='\t')   #zc_test_score
    df_1 = df_1.iloc[:400000]
    df_2 = df_2.iloc[:400000]
    fw=file(pathaa+'/innertrain_oot5_feautre','w')
    for fea in file(pathaa+'/real_inmodel'):
        fea = fea.strip('\r')
        fea = fea[:-1].split('\t')[0]
        fea = fea.strip('\r')
        if len(df_1[fea])>0 and len(df_2[fea])>0:
            if sum(np.isnan(df_1[fea]))==len(df_1[fea]) or sum(np.isnan(df_2[fea]))==len(df_2[fea]):
                fw.write(fea + '\t' + 'inf'+'\n')
            else:
                fw.write(fea + '\t' + str(round(psi(df_1[fea], df_2[fea]), 4))+'\n')
        else:
            fw.write(fea + '\t' + 'inf'+'\n')
