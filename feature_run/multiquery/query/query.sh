#!/bin/bash

source ~/.bashrc

arg_num=${#}
data=".data"
if [ $arg_num -eq 3 ];then
    uniq_id=$1
    day=$2
    inputsample=$3
fi

echo uniq_id is $uniq_id
echo day is $day
echo inputsample id $inputsample

bhive --hivevar uniq_id=${uniq_id} -f create_sample_table.sql

hadoop fs -put $inputsample /user/hive/warehouse/tj_tmp.db/feature_sample_${uniq_id}/

bhive --hivevar uniq_id=${uniq_id} --hivevar day=${day} -f get_sample_multiloan_query_v03.sql

hadoop fs -getmerge /user/hivewarehouse/tj_tmp.db/multiloan_query_v03_sample_${uniq_id} $inputsample$data

nohup python2.7 -u test_mapper.py $inputsample$data > cal_feature.log 2>&1 &

bhive --hivevar uniq_id=${uniq_id} -f drop_sample_table.sql
