# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

KS_PART = 10#20

def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = (head_pos + tail_pos) // 2
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)

def ks(y_true, y_prob, ks_part=KS_PART):
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
        if i == ks_part-1 or data[length*(i+1)//ks_part-1, 1] != data[length*(i+2)//ks_part-1, 1]:
            cut_list.append(data[length*(i+1)//ks_part-1, 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(data[length*(i+1)//ks_part-1, 1], data[:, 1], length*(i+1)//ks_part-1, length*(i+2)//ks_part-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            bad_num.append(sum(data[cut_pos_last+1:cut_pos+1, 0]))
            cut_pos_last = cut_pos

    order_num = np.array(order_num)
    bad_num = np.array(bad_num)

    good_num = order_num - bad_num
    order_ratio = np.array([round(x, 3) for x in order_num * 100 / float(length)])
    overdue_ratio = np.array([round(x, 3) for x in bad_num * 100 / [float(x) for x in order_num]])
    bad_ratio = np.array([round(sum(bad_num[:i+1])*100/float(sum_bad), 3) for i in range(len(bad_num))])
    good_ratio = np.array([round(sum(good_num[:i+1])*100/float(sum_good), 3) for i in range(len(good_num))])
    ks_list = abs(good_ratio - bad_ratio)
    ks = max(ks_list)

    try:
        span_list = ['[%.3f,%.3f]' % (min(data[:, 1]), round(cut_list[1], 3))]
        if len(cut_list) > 2:
            for i in range(2, len(cut_list)):
                span_list.append('(%.3f,%.3f]' % (round(cut_list[i-1], 3), round(cut_list[i], 3)))
    except:
        span_list = ['0']

    dic_ks = {
            'ks': ks,
            'span_list': span_list,
            'order_num': order_num,
            'bad_num': bad_num,
            'good_num': good_num,
            'order_ratio': order_ratio,
            'overdue_ratio': overdue_ratio,
            'bad_ratio': bad_ratio,
            'good_ratio': good_ratio,
            'ks_list': ks_list
            }

    return dic_ks

def ks_with_nil(y_true, y_prob, ks_part=KS_PART):
    data = np.vstack((y_true, y_prob)).T
    sort_ind = np.argsort(data[:, 1])
    data = data[sort_ind]

    length = len(y_prob)
    sum_bad = len([x for x in data[:, 0] if x == 1])
    sum_good = len([x for x in data[:, 0] if x == 0])
    sum_reject = len([x for x in data[:, 0] if x == -1])
    sum_accept = len([x for x in data[:, 0] if x == 0 or x == 1])

    cut_list = [0]
    order_num = []
    bad_num = []
    good_num = []
    accept_num = []
    reject_num = []

    cut_pos_last = -1
    for i in np.arange(ks_part):
        if i == ks_part-1 or data[length*(i+1)//ks_part-1, 1] != data[length*(i+2)//ks_part-1, 1]:
            cut_list.append(data[length*(i+1)//ks_part-1, 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(data[length*(i+1)//ks_part-1, 1], data[:, 1], length*(i+1)//ks_part-1, length*(i+2)//ks_part-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            bad_num.append(len([x for x in data[cut_pos_last+1:cut_pos+1, 0] if x == 1]))
            good_num.append(len([x for x in data[cut_pos_last+1:cut_pos+1, 0] if x == 0]))
            accept_num.append(len([x for x in data[cut_pos_last+1:cut_pos+1, 0] if x == 0 or x == 1]))
            reject_num.append(len([x for x in data[cut_pos_last+1:cut_pos+1, 0] if x == -1]))
            cut_pos_last = cut_pos

    order_num = np.array(order_num)
    bad_num = np.array(bad_num)
    good_num = np.array(good_num)
    accept_num = np.array(accept_num)
    reject_num = np.array(reject_num)

    #good_num = order_num - bad_num
    order_ratio = np.array([round(x, 3) for x in order_num * 100 / float(length)])
    #overdue_ratio = np.array([round(x, 3) for x in bad_num * 100 / [float(x) for x in order_num]])
    overdue_ratio = np.array([round(x, 3) for x in bad_num * 100 / [float(x) for x in accept_num]])
    bad_ratio = np.array([round(sum(bad_num[:i+1])*100/float(sum_bad), 3) for i in range(len(bad_num))])
    good_ratio = np.array([round(sum(good_num[:i+1])*100/float(sum_good), 3) for i in range(len(good_num))])
    accept_ratio = np.array([round(sum(accept_num[:i+1])*100/float(sum_accept), 3) for i in range(len(accept_num))])
    reject_ratio = np.array([round(sum(reject_num[:i+1])*100/float(sum_reject), 3) for i in range(len(reject_num))])
    ks_list = abs(good_ratio - bad_ratio)
    ks2_list = abs(good_ratio - reject_ratio)
    ks = max(ks_list)
    ks2 = max(ks2_list)

    try:
        span_list = ['[%.3f,%.3f]' % (min(data[:, 1]), round(cut_list[1], 3))]
        if len(cut_list) > 2:
            for i in range(2, len(cut_list)):
                span_list.append('(%.3f,%.3f]' % (round(cut_list[i-1], 3), round(cut_list[i], 3)))
    except:
        span_list = ['0']

    dic_ks = {
            'ks': ks,
            'ks2': ks2,
            'span_list': span_list,
            'order_num': order_num,
            'bad_num': bad_num,
            'good_num': good_num,
            'accept_num': accept_num,
            'reject_num': reject_num,
            'order_ratio': order_ratio,
            'overdue_ratio': overdue_ratio,
            'bad_ratio': bad_ratio,
            'good_ratio': good_ratio,
            'accept_ratio': accept_ratio,
            'reject_ratio': reject_ratio,
            'ks_list': ks_list,
            'ks2_list': ks2_list
            }

    return dic_ks

def print_ks(ks_info):
    print('ks = %.2f%%' % ks_info['ks'])
    print('\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量']))
    for i in range(len(ks_info['ks_list'])):
        print('%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i]))

def print_ks_with_nil(ks_info):
    print('ks = %.2f%%' % ks_info['ks'])
    print('ks2 = %.2f%%' % ks_info['ks2'])
    print('\t'.join(['seq', '评分区间', '订单数', '逾期数', '未逾数', '收单数', '拒单数', '订单比', '逾期率', '累坏比', '累好比', '累收比', '累拒比', 'KS', 'KS2']))
    for i in range(len(ks_info['ks_list'])):
        print('%d\t%s\t%d\t%d\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['accept_num'][i], ks_info['reject_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['accept_ratio'][i],  ks_info['reject_ratio'][i], ks_info['ks_list'][i],
            ks_info['ks2_list'][i]))
    print('Note: 订单数 = 收单数 + 拒单数')
    print('Note: 收单数 = 逾期数 + 未逾数')
    print('Note: KS     = 累好比 - 累坏比')
    print('Note: KS2    = 累好比 - 累拒比')


if __name__ == '__main__':
#    df_test = pd.read_csv('./model/prob_test', sep='\t')
#    dic_ks = ks(np.array(df_test['label']), np.array(df_test['prob']))
#    print_ks(dic_ks)
    yt = np.array([1,1,1,1,1,0,0,0,0,0])
    yp = np.array([0.9,0.8,0.7,0.6,0.55,0.43,0.32,0.33,0.22,0.12])
    print(len(yt))
    print(len(yp))
    print(yp[5])
    print(ks(yt,yp).get('ks'))
