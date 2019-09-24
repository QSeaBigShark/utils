
--开发人 wangzhenchao
--多头_快问

set mapred.job.name=tj_tmp.multi_loan3_qsearch_sample_${hivevar:uniq_id};  

-- 打开MapJoin自动优化，当前使用MapJoin Hint ( /*+ mapjoin(i)*/)无法生效，只能使用自动方式
set hive.auto.convert.join=true;                                                       
-- 大表小表的阀值 默认25000000 (25M)                                                   
set hive.mapjoin.smalltable.filesize=300000000;
set mapreduce.job.queuename=tj;
-- 如果存在小文件，增加以下参数
-- map join做group by 操作时，可以使用多大的内存来存储数据，如果数据太大，则不会保存在内存里，默认0.55
-- 每个Map最大输入大小，决定合并后的文件数
set mapred.max.split.size=512000000;
-- 一个节点上split的至少的大小 ，决定了多个data node上的文件是否需要合并
set mapred.min.split.size.per.node= 256000000;
-- 一个交换机下split的至少的大小，决定了多个交换机上的文件是否需要合并
set mapred.min.split.size.per.rack= 256000000;
-- 执行Map前进行小文件合并
set hive.input.format=org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;



--merge_sample
drop table if exists tj_tmp.multiloan_qsearch_v03_sample_${hivevar:uniq_id};
create table tj_tmp.multiloan_qsearch_v03_sample_${hivevar:uniq_id}
as
select
b.name,
b.encry_iden,
b.encry_mbl,
b.loan_dt,
b.uniq_id,
a.value
from tj_tmp.feature_sample_${hivevar:uniq_id} b --样本
join  (select * from tj_base.multi_loan3_qsearch_kv_d  where etl_dt = "${hivevar:day}")a
on a.mbl_num = b.encry_mbl
;
