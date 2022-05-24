#!/bin/bash

# Runs dbgen from separate script
# Was easier than spawning up a bunch of subprocesses  

TPCH_DIR='/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_code/tpch-tool' 
TPCH_TOOL_DIR='/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_code/tpch-tool/dbgen'

scale_factor=$1 

cd $TPCH_TOOL_DIR

# complete command-line args in /home/jw2027/tpch-tool/dbgen/README
./dbgen -s $scale_factor

# http://myfpgablog.blogspot.com/2016/08/tpc-h-queries-on-postgresql.html 
for i in `ls *.tbl`; do sed 's/|$//' $i > ${i/tbl/csv}; done;

# throw in /tmp/tables for ingestion into db
mkdir /tmp/tables;
rm *.tbl;
mv *.csv /tmp/tables;
