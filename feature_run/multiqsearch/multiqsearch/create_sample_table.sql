--bhive --hivevar uniq_id=zhangji01 -f create_sample_table.sql
drop table tj_tmp.feature_sample_multiqsearch_${uniq_id};
create EXTERNAL table tj_tmp.feature_sample_multiqsearch_${uniq_id}(
name varchar(256),
encry_iden varchar(256),
encry_mbl varchar(256),
loan_dt varchar(256),
uniq_id varchar(256)
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

