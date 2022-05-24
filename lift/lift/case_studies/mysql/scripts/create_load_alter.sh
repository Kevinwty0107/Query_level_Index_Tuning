#!/usr/bin/env bash
cd /local/scratch/mks40/dbgen
rm -rf *.tbl
./dbgen -s 1.0
mysql -u mks40 --password="" < /local/scratch/mks40/lift/lift/case_studies/mysql/scripts/create_table.sql
mysql -u mks40 --password="" < /local/scratch/mks40/lift/lift/case_studies/mysql/scripts/load_tbl_data.sql
mysql -u mks40 --password="" < /local/scratch/mks40/lift/lift/case_studies/mysql/scripts/alter_schema.sql