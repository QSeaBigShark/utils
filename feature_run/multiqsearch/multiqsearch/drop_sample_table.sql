--bhive --hivevar uniq_id=zhangji01 -f drop_sample_table.sql
drop table if exists tj_tmp.feature_sample_${uniq_id};

