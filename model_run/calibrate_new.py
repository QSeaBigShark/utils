import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

#import get_ks

percentage_list = []
for score in range(301,850):
	percentage_list.append(round(norm.cdf(score,575,68.75)*100, 4))
	#percentage_list.append(round( ,4))
np.savetxt('percentage_list.txt', percentage_list, fmt = '%.4f')


def get_src_quantiles(bad_prob_array, path):
	bad_prob_array = np.array(bad_prob_array)
	good_prob_array = 1 - bad_prob_array

	percentage_list = np.loadtxt('percentage_list.txt')
	src_quantiles = [0]
	for percent in percentage_list:
		src_quantiles.append(np.percentile(good_prob_array, percent))
	src_quantiles.append(1)
	np.savetxt(path + 'src_quantiles.txt', src_quantiles, fmt = '%.4f')


def binary_search(find, List):
	low = 0
	high = len(List)
	while low < high - 1:
		mid = (low + high)/2
		if find >= List[mid]:
			low = mid
		else:
			high = mid
	return low

def calibrate_new(bad_prob, src_quantiles):
	good_prob = 1- bad_prob
	if good_prob == 1:
		score = 850
	else:
		'''
		score = 300 + binary_search(good_prob, src_quantiles)
		'''
		for i in xrange(0,550):
			if good_prob >= src_quantiles[i] and good_prob < src_quantiles[i+1]:
				score = 300+i
				break
	return score

if __name__ == '__main__':
        path = r'/home/users/zhangji01.b/tx/data/hunhedata/v03/sample-score'
        #train = pd.read_table(path+'/train_select_10per_prob_', sep = '\t', header = 0)	
	#get_src_quantiles(train['prob'], path+r'/')
	
        test = pd.read_table(path+'/valid_select_10per_prob_', sep = '\t', header = 0)
	src_quantiles = np.loadtxt(path+r'/src_quantiles.txt')
	start_time = datetime.now()
	test['score'] = test['prob'].apply(lambda x: calibrate_new(x, src_quantiles))
	print test['score'].describe()
	cost_time = (datetime.now() - start_time).total_seconds()
	print cost_time
	test.to_csv(path+r'/valid_tsn_score',sep='\t', index = None)
