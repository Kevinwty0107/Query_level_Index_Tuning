import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..')) # /src for access to src/common (though not required for this file)

from imdb_common.imdb_util import imdb_tables, imdb_table_columns, imdb_table_columns_, column_type_operators, imdb_string_values, \
    random_float, sample_text, imdb_sample_fns
from imdb_common.sql_query import SQLQuery
from imdb_common.sql_workload import SQLWorkload

import numpy as np

def sort_queries(queries):
    """ Sort to promote index prefix intersection"""
    return sorted(queries, key=lambda query: len(query.query_cols), reverse=True)

class IMDBWorkload(SQLWorkload):
    
    def __init__(self, spec):
        """
        Encapsulates workload / workload generation from imdb 
        Based on imdb dbgen-derived data but not imdb qgen-derived queries 
        TODO: replace sampler strategy
        Boilerplated from lift/case_studies/{mongodb,mysql}/ 

        Args: 
            spec (dict): spec for workload; see controller for construction
        """
        
        self.tables = spec['tables'] # which relations to sample from?
        self.n_selections = spec['n_selections'] # how many attributes within those relations?
        self.scale_factor = spec['scale_factor'] # required for scaled_range sampling
        

    def generate_query_template(self, selectivities=None):
        """
        Sample a simple query like SELECT COUNT(*) FROM _ WHERE _ AND _ AND _
        """
        # sample a table for FROM clause
        tbl = np.random.choice(self.tables)
        query_string = "SELECT COUNT(*) FROM {} WHERE".format(tbl)
        selections = []
        # randomly select 1,2,..., or n_selections query attributes 
        # n_selections = np.random.randint(1, self.n_selections+1) 
        # TODO
        # workload 'engineering' again
        # justification? OLAP-y
        prob_delta = 2/(self.n_selections*(self.n_selections-1))
        prob = np.arange(0.0,2/(self.n_selections-1), prob_delta)
        n_selections = np.random.choice(np.arange(self.n_selections) + 1, p=prob)

        # sample columns from table for WHERE clause
        tbl_cols_ = imdb_table_columns_[tbl][0]  # TODO hack here. this is ordered.
        tbl_cols = imdb_table_columns[tbl] # this is not ordered, but contains column descriptors.

        # TODO 
        # workload 'engineering'
        # want queries to contain selective columns more frequently than non-selective columns
        # selectivites vary in scale (from 1e-7 to 1e-1), so picking in proportion will be too biased towards a few selective columns
        # exponentiating with exp < 1 will shift density from selective to non-selective columns
        # e.g. a command-line viz:
        # python3 -c 'exp = .2; import numpy as np; import matplotlib.pyplot as plt; labels=['L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY', 'L_LINENUMBER', 'L_QUANTITY', 'L_EXTENDEDPRICE', 'L_DISCOUNT', 'L_TAX', 'L_RETURNFLAG', 'L_LINESTATUS', 'L_SHIPDATE', 'L_COMMITDATE', 'L_RECEIPTDATE', 'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_COMMENT']; p = np.array([0.24994938524948698, 0.0333265846999316, 0.00166632923499658, 1.166430464497606e-06, 8.3316461749829e-06, 0.1556184872563306, 1.8329621584962379e-06, 1.499696311496922e-06, 4.998987704989739e-07, 3.3326584699931596e-07, 0.00042091476476013607, 0.0004109167893501566, 0.0004255804866181265, 6.665316939986319e-07, 1.166430464497606e-06, 0.7632899337884078]); p_ = (p ** exp) / np.sum(p ** exp); plt.figure(0); plt.bar(range(len(p)), p); plt.xticks(range(len(p)), labels, rotation=45, fontsize=5); plt.figure(1); plt.bar(range(len(p_)), p_); plt.xticks(range(len(p)), labels, rotation=45,fontsize=5); plt.show()'
        # for quick reference... [('L_ORDERKEY', 0.24994938524948698), ('L_PARTKEY', 0.0333265846999316), ('L_SUPPKEY', 0.00166632923499658), ('L_LINENUMBER', 1.166430464497606e-06), ('L_QUANTITY', 8.3316461749829e-06), ('L_EXTENDEDPRICE', 0.1556184872563306), ('L_DISCOUNT', 1.8329621584962379e-06), ('L_TAX', 1.499696311496922e-06), ('L_RETURNFLAG', 4.998987704989739e-07), ('L_LINESTATUS', 3.3326584699931596e-07), ('L_SHIPDATE', 0.00042091476476013607), ('L_COMMITDATE', 0.0004109167893501566), ('L_RECEIPTDATE', 0.0004255804866181265), ('L_SHIPINSTRUCT', 6.665316939986319e-07), ('L_SHIPMODE', 1.166430464497606e-06), ('L_COMMENT', 0.7632899337884078)]
        
        if selectivities is not None:
            T = 0.1
            tbl_2_col_2_sel = selectivities
            p = [tbl_2_col_2_sel[tbl][col] for col in tbl_cols_]   
            p = np.array(p)
            p = (p ** T) / np.sum(p ** T)
        else:
            p = None
        
        cols = np.random.choice(tbl_cols_, size=n_selections, replace=False, p=p)        
        
        tokens = []

        # sample operators
        for i in range(n_selections):
            col = cols[i]
            desc = tbl_cols[col]
            col_type = desc[0]
            col_type_ops = column_type_operators[col_type]

            # TODO workload 'engineering' again
            # = vs <, > makes index's effect more evident
            p = [0.5, 0.25, 0.25] if col_type is int or col_type is float else None
            col_op = np.random.choice(col_type_ops, p=p)
            
            tokens.append(col)
            tokens.append(col_op)

            selection = "{} {} '%s'".format(col, col_op)
            selections.append(selection)
        
        if n_selections == 1:
            selection_string = selection
        else:
            selection_string = " AND ".join(selections)
        query_string = "{} {}".format(query_string, selection_string)

        # sample operands (i.e. %s in above)
        def sample():
            sampled_args = []
            for col in cols:
                desc = tbl_cols[col]
                col_type, sample_type = desc[0], desc[1]

                sample = None
                if sample_type == "lookup":
                    sample = np.random.choice(imdb_string_values[col])
                elif sample_type == "fixed_range":
                    range_tuple = desc[2]
                    if col_type == int:
                        sample = np.random.randint(low=range_tuple[0], high=range_tuple[1])
                    elif col_type == float:
                        sample = random_float(low=range_tuple[0], high=range_tuple[1])
                elif sample_type == "scaled_range":
                    range_tuple = desc[2]
                    scaled_low = range_tuple[0] * self.scale_factor
                    scaled_high = range_tuple[1] * self.scale_factor
                    if col_type == int:
                        sample = np.random.randint(low=scaled_low, high=scaled_high)
                    elif col_type == float:
                        sample = random_float(low=scaled_low, high=scaled_high)
                elif sample_type == "text":
                    sample = sample_text()
                elif sample_type == "sample_fn":
                    sample = imdb_sample_fns[col]()
                elif sample_type == "scaled_sample_fn":
                    sample = imdb_sample_fns[col](self.scale_factor)
                elif sample_type == "sample_fn_k":
                    sample = imdb_sample_fns[col](1)
                else:
                    raise ValueError("No arg sampled for {} with spec {}".format(col, desc))

                sampled_args.append(sample)

            return tuple(sampled_args)
        
        return SQLQuery(query_string, query_tbl=tbl, query_cols=cols, 
                        sample_fn=sample, tokens=tokens)

    
    def define_demo_queries(self, n_queries):
        """

        Returns:
            list of SQLQuery: queries encapsulated in SQLQuery query
        """

        pass

    def define_train_queries(self, n_queries):

        return [self.generate_query_template() for _ in range(n_queries)]

    def define_test_queries(self, n_queries):

        return [self.generate_query_template() for _ in range(n_queries)]

    def query_from_csv(self, query_csv):
        """
        Restores a serialized representation of a SQLQuery object.
        
        Args:
            query_csv (list): tighly coupled to sql_query::SQLQuery::as_csv_row
                              [self.query_string, self.query_cols, self.query_tbl, index_cols, self.tokens]
        Returns:
            SQLQuery
        """

        query_cols = query_csv[1].split(",")
        query_tbl = query_csv[2]
        index_columns = None if query_csv[3] == "[]" else query_csv[3].split(",")
        tokens = query_csv[4].split(",")
        query_string = "SELECT COUNT(*) FROM {} WHERE".format(query_tbl)

        # operand (attribute operand) + operator
        selections = []
        for i in range(len(query_cols)):
            selection = "{} {} '%s'".format(query_cols[i], tokens[i*2+1])
            selections.append(selection)

        if len(selections) == 1: 
            selection_string = selection
        else:
            selection_string = " AND ".join(selections)
        query_string = "{} {}".format(query_string, selection_string)
        query = SQLQuery(query_string, query_tbl=query_tbl, index_cols=index_columns,
                         query_cols=query_cols, tokens=tokens)


        query = SQLQuery(query_string, query_tbl=query_tbl, query_cols=query_cols, 
                        index_cols=index_columns, tokens=tokens)

        # operands (attribute value operands)
        # TODO refactor 
        def sample():
            sampled_args = []
            for col in query_cols:
                tbl_cols = imdb_table_columns[query_tbl] # add
                desc = tbl_cols[col] # add
                col_type, sample_type = desc[0], desc[1]

                sample = None
                if sample_type == "lookup":
                    sample = np.random.choice(imdb_string_values[col])
                elif sample_type == "fixed_range":
                    range_tuple = desc[2]
                    if col_type == int:
                        sample = np.random.randint(low=range_tuple[0], high=range_tuple[1])
                    elif col_type == float:
                        sample = random_float(low=range_tuple[0], high=range_tuple[1])
                elif sample_type == "scaled_range":
                    range_tuple = desc[2]
                    scaled_low = range_tuple[0] * self.scale_factor
                    scaled_high = range_tuple[1] * self.scale_factor
                    if col_type == int:
                        sample = np.random.randint(low=scaled_low, high=scaled_high)
                    elif col_type == float:
                        sample = random_float(low=scaled_low, high=scaled_high)
                elif sample_type == "text":
                    sample = sample_text()
                elif sample_type == "sample_fn":
                    sample = imdb_sample_fns[col]()
                elif sample_type == "scaled_sample_fn":
                    sample = imdb_sample_fns[col](self.scale_factor)
                elif sample_type == "sample_fn_k":
                    sample = imdb_sample_fns[col](1)
                else:
                    raise ValueError("No arg sampled for {} with spec {}".format(col, desc))
                    
                sampled_args.append(sample)
            return tuple(sampled_args)

        query.sample_fn=sample
        return query
