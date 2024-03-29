# -*- coding:utf-8 -*-

#########################################################
# author: zhouchujie
# update_dt: 2019-02-27
#########################################################
# 计算ks
#########################################################

import pandas as pd
import numpy as np
import math

# 分箱个数
KS_PART = 10


def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = (head_pos + tail_pos) / 2
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)


def ks2(y_true, y_prob, ks_part=KS_PART, dec_pos=3):
    data = np.vstack((y_true, y_prob)).T
    sort_ind = np.argsort(data[:, 1])
    data = data[sort_ind]

    length = len(y_prob)
    sum_bad = sum(data[:, 0])
    sum_good = length - sum_bad

    cut_list = [0]
    order_num = []
    bad_num = []

    cut_pos_last = -1
    for i in np.arange(ks_part):
        if i == ks_part-1 or data[length*(i+1)/ks_part-1, 1] != data[length*(i+2)/ks_part-1, 1]:
            cut_list.append(data[length*(i+1)/ks_part-1, 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(data[length*(i+1)/ks_part-1, 1], data[:, 1], length*(i+1)/ks_part-1, length*(i+2)/ks_part-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            bad_num.append(sum(data[cut_pos_last+1:cut_pos+1, 0]))
            cut_pos_last = cut_pos

    order_num = np.array(order_num)
    bad_num = np.array(bad_num)

    good_num = order_num - bad_num
    order_ratio = np.array([x for x in order_num * 100 / float(length)])
    overdue_ratio = np.array([x for x in bad_num * 100 / [float(x) for x in order_num]])
    bad_ratio_sum = np.array([sum(bad_num[:i+1])*100/float(sum_bad) for i in range(len(bad_num))])
    good_ratio_sum = np.array([sum(good_num[:i+1])*100/float(sum_good) for i in range(len(good_num))])
    ks_list = abs(good_ratio_sum - bad_ratio_sum)
    ks = max(ks_list)

    bad_ratio = bad_num / float(sum_bad)
    good_ratio = good_num / float(sum_good)
    woe = map(lambda x: 0 if x == 0 else math.log(x), bad_ratio / good_ratio)
    iv_list = (bad_ratio - good_ratio) * woe
    iv = sum(iv_list)

    #print cut_list

    try:
        if dec_pos == 0:
            span_list = ['[%d,%d]' % (int(round(min(data[:, 1]), dec_pos)), int(round(cut_list[1], dec_pos)))]
        else:
            span_list = ['[%s,%s]' % (round(min(data[:, 1]), dec_pos), round(cut_list[1], dec_pos))]
        if len(cut_list) > 2:
            for i in range(2, len(cut_list)):
                if dec_pos == 0:
                    span_list.append('(%d,%d]' % (int(round(cut_list[i-1], dec_pos)), int(round(cut_list[i], dec_pos))))
                else:
                    span_list.append('(%s,%s]' % (round(cut_list[i-1], dec_pos), round(cut_list[i], dec_pos)))

    except:
        span_list = ['0']

    dic_ks = {
            'iv': iv,
            'ks': ks,
            'span_list': span_list,
            'order_num': order_num,
            'bad_num': bad_num,
            'good_num': good_num,
            'order_ratio': order_ratio,
            'overdue_ratio': overdue_ratio,
            'bad_ratio': bad_ratio_sum,
            'good_ratio': good_ratio_sum,
            'ks_list': ks_list,
            'woe': woe,
            'iv_list': iv_list
            }

    return dic_ks


def ks(y_true, y_prob, miss_order_num, miss_ovd_num, ks_part=KS_PART, dec_pos=3):

    data = np.vstack((y_true, y_prob)).T
    sort_ind = np.argsort(data[:, 1])
    data = data[sort_ind]  # 预测值升序排序

    length = len(y_prob)  # 命中样本总数
    sum_bad = sum(data[:, 0])  # 命中坏样本总数
    sum_good = length - sum_bad  # 命中好样本总数

    length_all = length + miss_order_num  # 样本总数
    sum_bad_all = sum_bad + miss_ovd_num  # 坏样本总数
    sum_good_all = length_all - sum_bad_all  # 好样本总数

    cut_list = [0]
    order_num = []
    bad_num = []

    cut_pos_last = -1
    for i in np.arange(ks_part):
        if i == ks_part-1 or data[length*(i+1)/ks_part-1, 1] != data[length*(i+2)/ks_part-1, 1]:
            cut_list.append(data[length*(i+1)/ks_part-1, 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(
                    data[length*(i+1)/ks_part-1, 1],
                    data[:, 1],
                    length*(i+1)/ks_part-1,
                    length*(i+2)/ks_part-2
                )    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            bad_num.append(sum(data[cut_pos_last+1:cut_pos+1, 0]))
            cut_pos_last = cut_pos

    order_num = np.array(order_num)
    bad_num = np.array(bad_num)

    good_num = order_num - bad_num
    # order_ratio = np.array([x for x in order_num * 100 / float(length)])
    overdue_ratio = np.array([x for x in bad_num * 100 / [float(x) for x in order_num]])
    bad_ratio_sum = np.array([sum(bad_num[:i+1])*100/float(sum_bad) for i in range(len(bad_num))])
    good_ratio_sum = np.array([sum(good_num[:i+1])*100/float(sum_good) for i in range(len(good_num))])
    ks_list = abs(good_ratio_sum - bad_ratio_sum)
    ks = max(ks_list)

    bad_num_iv = list(bad_num)
    bad_num_iv.insert(0, miss_ovd_num)
    good_num_iv = list(good_num)
    good_num_iv.insert(0, miss_order_num-miss_ovd_num)
    bad_num_iv = np.array(bad_num_iv)
    good_num_iv = np.array(good_num_iv)
    bad_ratio = bad_num_iv / float(sum_bad_all)
    good_ratio = good_num_iv / float(sum_good_all)
    woe = map(lambda x: 0 if x == 0 else math.log(x), bad_ratio / good_ratio)
    iv_list = (bad_ratio - good_ratio) * woe
    iv_list = list(iv_list)
    iv = sum(iv_list)
    woe = list(woe)

    try:
        if dec_pos == 0:
            span_list = ['[%d,%d]' % (int(round(min(data[:, 1]), dec_pos)), int(round(cut_list[1], dec_pos)))]
        else:
            span_list = ['[%s,%s]' % (round(min(data[:, 1]), dec_pos), round(cut_list[1], dec_pos))]
        if len(cut_list) > 2:
            for i in range(2, len(cut_list)):
                if dec_pos == 0:
                    span_list.append('(%d,%d]' % (int(round(cut_list[i-1], dec_pos)), int(round(cut_list[i], dec_pos))))
                else:
                    span_list.append('(%s,%s]' % (round(cut_list[i-1], dec_pos), round(cut_list[i], dec_pos)))

    except:
        span_list = ['0']
    span_list.insert(0, 'Nan')
    order_num = list(order_num)
    bad_num = list(bad_num)
    good_num = list(good_num)
    order_num.insert(0, miss_order_num)
    bad_num.insert(0, miss_ovd_num)
    good_num.insert(0, miss_order_num - miss_ovd_num)
    overdue_ratio = list(overdue_ratio)
    overdue_ratio_na = 0 if miss_order_num == 0 else miss_ovd_num * 100 / miss_order_num
    overdue_ratio.insert(0, overdue_ratio_na)
    order_ratio_all = [x * 100 / float(length_all) for x in order_num]
    bad_ratio_sum = list(bad_ratio_sum)
    bad_ratio_sum.insert(0, np.nan)
    good_ratio_sum = list(good_ratio_sum)
    good_ratio_sum.insert(0, np.nan)
    ks_list = list(ks_list)
    ks_list.insert(0, np.nan)

    dic_ks = {
            'iv': iv,
            'ks': ks,
            'span_list': span_list,
            'order_num': order_num,
            'bad_num': bad_num,
            'good_num': good_num,
            'order_ratio': order_ratio_all,
            'overdue_ratio': overdue_ratio,
            'bad_ratio': bad_ratio_sum,
            'good_ratio': good_ratio_sum,
            'ks_list': ks_list,
            'woe': woe,
            'iv_list': iv_list
            }

    return dic_ks

def print_ks(ks_info):
    print 'iv\t%.4f' % ks_info['iv']
    print 'ks\t%.2f%%' % ks_info['ks']
    print '\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量', 'WOE', 'IV统计量'])
    for i in range(len(ks_info['ks_list'])):
        print '%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.4f\t%.4f' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i], ks_info['woe'][i], ks_info['iv_list'][i])

def write_ks(ks_info, fout):
    fout.write('iv\t%.4f\n' % ks_info['iv'])
    fout.write('ks\t%.2f%%\n' % ks_info['ks'])
    fout.write('\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量', 'WOE', 'IV统计量'])+'\n')
    for i in range(len(ks_info['ks_list'])):
        fout.write('%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.4f\t%.4f\n' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i], ks_info['woe'][i], ks_info['iv_list'][i]))


if __name__ == '__main__':
    df_test = pd.read_csv('tpd_check.txt', sep='\t')
    #df_test = df_test[(df_test['loan_dt']>='2018-06-01')]
    #df_test["label"] = df_test["label"].astype(float)
    dic_ks = ks(np.array(df_test['label']), np.array(df_test['score_autotest']))
    print_ks(dic_ks)
