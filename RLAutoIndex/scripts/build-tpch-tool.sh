#!/bin/bash

# Retrieves and builds tpch tools dbgen and qgen

#TPCH_DIR='/local/scratch/jw2027/tpch-tool' 
#TPCH_TOOL_DIR='/local/scratch/jw2027/tpch-tool/dbgen'

TPCH_DIR='/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_code/tpch-tool' 
TPCH_TOOL_DIR='/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_code/tpch-tool/dbgen'

git clone https://github.com/jerwelborn/tpch-tool.git $TPCH_DIR

cd $TPCH_TOOL_DIR
cp makefile.suite Makefile;
sed -i '' '103s/$/gcc/' Makefile;
sed -i '' '109s/$/ORACLE/' Makefile;
sed -i '' '110s/$/LINUX/' Makefile;
sed -i '' '111s/$/TPCH/' Makefile;
make;
