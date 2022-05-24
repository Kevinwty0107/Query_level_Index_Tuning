from imdb_workload import IMDBWorkload
from imdb_common.sql_query import SQLQuery
from imdb_common.sql_workload import SQLWorkload
from common.tpch_workload import TPCHWorkload

n_selections = 3
spec ={"tables":['lineitem'],"scale_factor": 1, # TODO specify this somewhere, not just in imdb_util 
    "n_selections": n_selections} 
workload = TPCHWorkload(spec) 
query = workload.generate_query_template()
print(query.as_csv_row())
spec ={"tables":['title'],"scale_factor": 1, # TODO specify this somewhere, not just in imdb_util 
    "n_selections": n_selections} 
workload = IMDBWorkload(spec) 
query = workload.generate_query_template()
print(query.as_csv_row())