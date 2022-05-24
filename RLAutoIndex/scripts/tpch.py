#!/usr/bin/python3

"""

    utility script for different tasks with tpch database 
    assumes you've run setup db script and added a super-user user

    Todos
        - hardcoded for jw2027
        - create queries from query templates

    References
        - http://www.tpc.org/tpc_documents_current_versions/current_specifications.asp
        - http://myfpgablog.blogspot.com/2016/08/tpc-h-queries-on-postgresql.html
"""

import argparse
import numpy as np
import os
import re
import subprocess
import sys
import time

import psycopg2 as pg
from tpch_util import TPCH_DIR, TPCH_TOOL_DIR, DATA_DIR, DSN, TPCH_DSN

class TPCHClient():
    """Encapsulates 2 cxns, one to a non-tpch default db and another to the tpch db. 

        Comments:
        - python psycopg driver has transaction-specific semantics that are handled cleanly in context managers. 
          I relied on cxn.set_session(autocommit=True), which ignores these implicit semantics, because of how I handled cxns.
        - The exception handling here is extremely brittle 

    """

    def __init__(self):
        self.cxn = self.__connect(DSN)
        try: 
            self.tpch_cxn = self.__connect(TPCH_DSN)
        except pg.OperationalError as e:
            self.tpch_cxn = None 
    
    def __connect(self, DSN):
        cxn = pg.connect(DSN)
        cxn.set_session(autocommit=True)
        return cxn

    def repopulate(self, scale_factor):
        """dbgen to generate data, build-tpch-tables.sql to create tables and copy data into tables

        Args:
            scale_factor (float) : -s, --scale_factor arg to dbgen, only certain values are TPCH compliant, but scale of 1 corresponds to 1 GB.

        """
        
        # drop db if exists, this isn't terribly expensive
        print('dropping db...')
        if self.tpch_cxn:
            self.tpch_cxn.close()
            with self.cxn.cursor() as cur:
                cur.execute('DROP DATABASE tpch')
            
        # create db, connection to db
        print('creating db... connecting to db...')
        with self.cxn.cursor() as cur:
            cur.execute('CREATE DATABASE tpch')
        self.tpch_cxn = self.__connect(TPCH_DSN)

        # generate data
        print('running dbgen... repopulating db...')
        tic=time.time()
        subprocess.run(['./dbgen.sh', str(scale_factor)]) # TODO whoops put absolute path

        # create tables, copy data into tables
        with open(os.path.join(TPCH_DIR, 'build-tpch-tables.sql'), 'r') as f, \
            self.tpch_cxn.cursor() as cur:

            cur.execute(f.read())
        
        toc = time.time()
        print('...took {} seconds'.format(round(toc-tic)))

        # rm generated data
        print('cleaning up...')
        subprocess.run(['rm', '-rf', DATA_DIR])

        print('saving scale_factor to tpch_sf.txt')
        np.savetxt('../../tpch_sf.txt', np.asarray([scale_factor]))
    
    def close(self):
        self.cxn.close()
        self.tpch_cxn.close()

    def get_indices(self):
        """
            See system catalog chapter in docs
            Additional index summaries, statistics: https://wiki.postgresql.org/wiki/Index_Maintenance

        """

        def query(rel): 
            return """SELECT pg_class.relname, pg_index.indkey
                    FROM pg_class, pg_index
                    WHERE (pg_index.indexrelid = pg_class.oid)
                    AND (pg_index.indrelid = (SELECT pg_class.oid FROM pg_class WHERE pg_class.relname = \'{}\'));
                    """.format(rel)

        rels = tpch.schema.keys()
        idxs = dict.fromkeys(rels)

        with self.tpch_cxn.cursor() as curs:
            for rel in rels:
                curs.execute(query(rel))
                idxs_ = curs.fetchall()
                idxs_ = dict(idxs_) # index -> index keys  
            
                # TODO this can be done cleanly in query
                # pg_index.indkey is a SQL array of attributes indices in their respective tables
                split=lambda attrs: attrs.split() 
                cast=lambda attrs: list(map(lambda attr: int(attr)-1, attrs))
                invertindex=lambda attrs: list(np.array(schema[rel])[attrs])

                attrs = idxs_.values() 
                attrs = list(map(split, attrs))
                attrs = list(map(cast, attrs))
                attrs = list(map(invertindex, attrs))

                idxs_ = {key : attrs[i] for i, key in enumerate(idxs_.keys())}
                idxs[rel] = idxs_
        return idxs 

    def set_index(self, idx, rel, attrs):
        """
        TODO make query more extensible, builds complete b-trees only.
        """

        query = 'CREATE INDEX {} ON {} ({})'.format(idx, rel, ','.join(attrs))

        with self.tpch_cxn.cursor() as curs:
            try:
                curs.execute(query)
            except pg.ProgrammingError as e:
                print(e)
    


##
# script utils
#
def parser():
    parser = argparse.ArgumentParser(description='utility script for different tasks with tpch database')
    parser.add_argument('-r', '--repopulate', action='store_true', help='')
    parser.add_argument('-s', '--scale_factor', required='-r' in sys.argv or '--repopulate' in sys.argv, action='store', type=float)
    return parser 

def main():

    args = parser().parse_args()

    tpch = TPCHClient()    

    if args.repopulate:
        tpch.repopulate(scale_factor=args.scale_factor)

if __name__ == "__main__":
    main()
