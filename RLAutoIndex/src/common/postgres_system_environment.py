# TODO remove this
import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.rl_model.system_environment import SystemEnvironment
sys.path.insert(0, os.path.join(head, '..')) # /src for access to src/common (though not required for this file)
from common.tpch_util import tpch_tables, tpch_table_columns

import time
if sys.version_info[0] == 2: # opentuner runs 2 not 3
    time.monotonic = time.time 

import enum
import os
import logging 
import psycopg2 as pg
import re




# TODO stick somewhere else
TPCH_DIR='~/Desktop/research/Multi_level_Index_Tuning/tpch-tool' 
TPCH_TOOL_DIR = os.path.join(TPCH_DIR, 'dbgen')
DATA_DIR = '/tmp/tables'
DSN = "dbname=postgres user=tw557"
TPCH_DSN = "dbname=tpch user=tw557"

class Action(enum.IntEnum):
    noop, duplicate_index, index = 0, 1, 2

class PostgresSystemEnvironment(SystemEnvironment):
    """
        Encapsulates environment

        N.b. agent indices have '_42'


    """

    def __init__(self, tbls):
        """
        """
        self.tbls = tbls

        self.logger = logging.getLogger(__name__)

        self.cxn = self.__connect(DSN)
        try: 
            self.tpch_cxn = self.__connect(TPCH_DSN)
        except pg.OperationalError as e:
            self.tpch_cxn = None 

        self.index_set = set()
        self.tbl_2_col_2_sel ={}

    def __connect(self, DSN):
        cxn = pg.connect(DSN)
        cxn.set_session(autocommit=True)
        return cxn
    
    def close(self):
        self.cxn.close()
        self.tpch_cxn.close()

    def act(self, action):
        """
        Creates compound index, as advised by agent

        Args:
            action (dict): contains cols, table containing cols for index
        """
        
        action_type = None

        cols, tbl = action['index'], action['table']
        index = '_'.join(cols) + '_42'
        if index in self.index_set:
            start = time.monotonic()
            self.logger.info('action cannot complete (index already in index set)')
            action_type = Action.duplicate_index.name
            act_time = time.monotonic()-start
        elif cols == []:
            start = time.monotonic()
            self.logger.info('action cannot complete (is no-op)') 
            action_type = Action.noop.name
            act_time = time.monotonic()-start
        else:
            with self.tpch_cxn.cursor() as curs:
                try:
                    start = time.monotonic() #TODO: Adjustment on the idx creation time calculation
                    self.logger.info("creating compound index %s on %s" % (index, tbl))
                    curs.execute("CREATE INDEX %s ON %s (%s)" %
                                (index, tbl, ','.join(cols)))            
        
                    self.index_set.add(index)
                    act_time = time.monotonic()-start
                    action_type = Action.index.name
                except pg.Error as e:
                    start = time.monotonic()
                    print(e)
                    act_time = time.monotonic()-start
                    action_type = Action.noop.name

        return act_time, action_type
    
    def execute(self, query, explain=True):
        
        """
        Having created compound index, executes query and returns runtime

        Args:
            query
            explain (bool): run with EXPLAIN ANALYZE, return indices used if any
        """

        query_string, query_string_args = query.sample_query()
        query_string = query_string % query_string_args
        
        if explain:
            query_string = 'EXPLAIN ANALYZE ' + query_string

        runtime = None
        try: 
            with self.tpch_cxn.cursor() as curs:
                start = time.monotonic()
                curs.execute(query_string)
                runtime = time.monotonic() - start
                res = curs.fetchall() # TODO 
        except pg.Error as e:
            print(e)

        idx, idx_str = [], ''
        if explain:
            for tup in res:
                if "Index Cond" in tup[0]:
                    # searching for something like Index Cond: (l_quantity < '4'::numeric)
                    idx_str = tup[0]
                    break # assume only one
            if idx_str != '':
                # idx = [col for col in query.query_cols if col in idx_str] 
                idx = re.findall('\([a-zA-Z_]+', idx_str)
                idx = list(map(lambda s: s[1:].upper(), idx))

        return runtime, idx

    def system_status(self):
        """
        Compute size of index set
        There are a few approaches for this:
            - the psql command / meta-command \di+ summarizes what we want. Starting psql with psql -E exposes the SQL.
            - see scripts/tpch.py for what I was employing earlier

        Here's how result set is returned:

             Name      |  Table   |   Size
        ---------------+----------+----------
         part_pkey     | part     |  2260992
         region_pkey   | region   |    16384
        ...
        
        [('part_pkey', 'part', 2260992), ('region_pkey', 'region', 16384), ...
        """

        query = """SELECT c.relname as "Name",
                          c2.relname as "Table",
                          pg_catalog.pg_table_size(c.oid) as "Size"
                   FROM pg_catalog.pg_class c
                        LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        LEFT JOIN pg_catalog.pg_index i ON i.indexrelid = c.oid
                        LEFT JOIN pg_catalog.pg_class c2 ON i.indrelid = c2.oid
                    WHERE c.relkind IN ('i','')
                        AND n.nspname <> 'pg_catalog'
                        AND n.nspname <> 'information_schema'
                        AND n.nspname !~ '^pg_toast'
                        AND pg_catalog.pg_table_is_visible(c.oid);"""
        
        
        with self.tpch_cxn.cursor() as curs:
            curs.execute(query)
            res = curs.fetchall()

        index_set_size = 0.0
        for row in res:
            # if '_42' in row[0]: index_set_size += row[2] 
            index_set_size += row[2] 
        index_set_size /= 1024*1024 
        
        return index_set_size, self.index_set


    def reset(self):
        """
           Remove all agent-initiated indices 

            schemaname | tablename |   indexname   | tablespace |
           ------------+-----------+---------------+------------+
            public     | customer  | customer_pkey |            |
        """

        tbl_idxs = 'SELECT * FROM pg_indexes WHERE tablename = \'%s\''
        
        for tbl in tpch_tables: # TODO self.tbls would suffice
        
            with self.tpch_cxn.cursor() as curs:
                curs.execute(tbl_idxs % tbl)
                idxs = curs.fetchall()
    
                for idx in idxs:
                    if '_42' in idx[2]:
                        curs.execute('DROP INDEX %s' % idx[2])

        self.index_set.clear()


    def compute_column_selectivity(self):

        rows_query = 'SELECT COUNT(*) FROM \"%s\";'
        distinct_rows_query = 'SELECT COUNT(DISTINCT %s) FROM \"%s\";'  
        tbl_2_col_2_sel = {}

        with self.tpch_cxn.cursor() as curs:
            for tbl in self.tbls: 
                col_2_sel = {}
                curs.execute(rows_query % tbl)
                total = curs.fetchall()[0][0]

                for col in tpch_table_columns[tbl].keys():
                    curs.execute(distinct_rows_query % (col, tbl))
                    distinct_total = curs.fetchall()[0][0]

                    col_2_sel[col] = distinct_total / total
                
                tbl_2_col_2_sel[tbl] = col_2_sel

        self.tbl_2_col_2_sel = tbl_2_col_2_sel
    
