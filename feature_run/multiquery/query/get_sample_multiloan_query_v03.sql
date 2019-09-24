--开发人 李任重	
--多头机构表

set mapred.job.name=tj_tmp.multi_loan_query_v03_sample_${hivevar:uniq_id};          
set hive.auto.convert.join=true;       
set hive.mapjoin.smalltable.filesize=300000000;
set mapreduce.job.queuename=tj;
set mapred.max.split.size=512000000;
set mapred.min.split.size.per.node= 256000000;
set mapred.min.split.size.per.rack= 256000000;
set hive.input.format=org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
set mapreduce.map.memory.mb=8192;
set mapreduce.reduce.memory.mb=8192;
set mapred.child.java.opts="-Xmx8192m";

drop table if exists tj_tmp.multiloan_query_v03_sample_tmp_${hivevar:uniq_id};
create table tj_tmp.multiloan_query_v03_sample_tmp_${hivevar:uniq_id}
as
select
b.name,
b.encry_iden,
b.encry_mbl,
b.loan_dt,
b.uniq_id,
a.tj_user_id,
a.tag_type,
a.tj_resource_id,
a.resource_type_id,
a.create_tm,
a.create_dt,
a.update_dt,
a.query_price,
a.query_res,
a.is_dirty_data
from tj_tmp.feature_sample_${hivevar:uniq_id} b  --样本
join (select * from tj_base.multi_loan3_query_st_d where etl_dt <= "${hivevar:day}")a
on a.query_mbl_num = b.encry_mbl
where  a.create_dt>=date_sub(b.loan_dt,360) and a.create_dt < b.loan_dt
;


--加工成KV表
drop table if exists tj_tmp.multiloan_query_v03_sample_${hivevar:uniq_id};
create table tj_tmp.multiloan_query_v03_sample_${hivevar:uniq_id}
as
select
name,
encry_iden,
encry_mbl,
loan_dt,
uniq_id,
concat_ws('\005',
    collect_list(
    concat_ws('\004',
        array(
               nvl(cast(tj_user_id as string),'') ,
               nvl(cast(tag_type as string),'') ,
               nvl(cast(tj_resource_id as string),'') ,
               nvl(cast(resource_type_id as string),'') ,
               nvl(cast(create_tm as string),'') ,
               nvl(cast(create_dt as string),'') ,
               nvl(cast(update_dt as string),'') ,
               nvl(cast(query_price as string),'') ,
               nvl(cast(query_res as string),'') ,
               nvl(cast(is_dirty_data as string),'') )))) value
from tj_tmp.multiloan_query_v03_sample_tmp_${hivevar:uniq_id}
group by name, encry_iden, encry_mbl, loan_dt, uniq_id
;
