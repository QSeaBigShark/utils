# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys
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
    except Exception:
        #logging.warn('psi not supported')
        psi = -1
    return psi
key = ['name','idcard','phone','loan_dt','label','month','uniq_id']
def compute_df_psi(df_1,df_2,month_prefix):
    list1 = df_1.columns.tolist()
    list2 = df_2.columns.tolist()
    list_res = list(set(list1).intersection(set(list2)))
    list_res = list(set(list_res) - set(key))
    #print list_res
    fw=file(str(month_prefix)+'.psi','w')
    fw.write("feature_name" + '\t' + 'psi'+'\n')

    for item in list_res:
        if len(df_1[item])>0 and len(df_2[item])>0:
            if sum(np.isnan(df_1[item]))==len(df_1[item]) or sum(np.isnan(df_2[item]))==len(df_2[item]):
                fw.write(item + '\t' + 'inf'+'\n')
            else:
                fw.write(item + '\t' + str(round(psi(df_1[item], df_2[item]), 4))+'\n')
        else:
            fw.write(item + '\t' + 'inf'+'\n')
def month_diff(month_1,month_2):
    m1 = month_1.split('-')
    m2 = month_2.split('-')
    return 12*(int(m2[0]) - int(m1[0])) + int(m2[1]) - int(m1[1])

def save_month_psi(df,month_gap):
    df['loan_dt'] = df['loan_dt'].astype("str")
    df['month'] = df['loan_dt'].apply(lambda x: x[0:7])
    month_list = sorted(df['month'].unique())
    length = len(month_list)
    for i in range(length - 1):
        for j in range(i+1,length):
            month1 = month_list[i]
            month2 = month_list[j]
            diff = month_diff(month1,month2)
            if diff == month_gap:
                df1 = df[df['month'] == month1]
                df2 = df[df['month'] == month2]
                compute_df_psi(df1,df2,month1 + '~' + month2)
                continue
    
if __name__ == '__main__':
    
    path_prefix = r'/home/users/zhangji01.b/tx/data/innerdata/month/'
    df_file = sys.argv[1]
    month_gap = sys.argv[2]
    df = pd.read_csv(path_prefix+df_file, sep='\t')
    save_month_psi(df,int(month_gap))
