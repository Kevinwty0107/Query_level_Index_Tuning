from lift.rl_model.system_environment import SystemEnvironment
from lift.util.exec_util import local_exec
import time
import logging
import MySQLdb


class MySQLSystemEnvironment(SystemEnvironment):

    def __init__(self, user="mks40", db="tpch", all_tables=None):
        """
        Creates an environment to create index_set.

        Some client examples:

        http://mysql-python.sourceforge.net/MySQLdb.html#some-examples

        Args:
            user (str): MySQL user. This user must exist in the database. See setup scripts.
        """
        self.logger = logging.getLogger(__name__)
        self.user = user
        self.db = db

        # Names of indices.
        self.index_set = set()
        self.all_tables = all_tables
        self.all_indices = {}
        self.connection = None
        self.connect()

    def connect(self):
        self.connection = MySQLdb.connect(db=self.db, user=self.user, passwd="", host="127.0.0.1")

    def act(self, action):
        assert isinstance(action, dict)
        start = time.monotonic()
        # self.logger.info("Action is = {}".format(action))
        if action:
            index_to_create = action["index"]
            index_name = "_".join(["{}_{}_1".format(index_tuple[0], index_tuple[1]) for index_tuple in index_to_create])
            if index_name in self.index_set:
                self.logger.info("Action {} already in index set, not executing.".format(index_to_create))
            elif self.is_noop(index_to_create):
                self.logger.info("Action {} is no-op, not executing.".format(index_to_create))
            else:
                # CREATE INDEX index_name ON table_name (column_1,column_2);
                tuple_with_sort_orders = ["{} {}".format(index_tuple[0], index_tuple[1]) for index_tuple in index_to_create]
                index_string = ",".join(tuple_with_sort_orders)
                table = action["table"]
                try:
                    cursor = self.connection.cursor()
                    # self.logger.info("Creating index {} on table {}.".format(index_name, table))
                    cursor.execute("CREATE INDEX %s ON %s (%s)" % (index_name, table, index_string))
                    self.index_set.add(index_name)
                    self.all_indices[index_name] = action
                    cursor.close()
                except Exception:
                    self.logger.info("Index {} could not be created. Index was in index set: {}.".format(
                        index_name, str(index_name in self.index_set)))

        return time.monotonic() - start

    def execute(self, query_string, query_args=None):
        """
        Executes query, returns runtime.

        Args:
            query_string (str): Fully formed SQL string.
            query_args(tuple): Optional tuple of string arguments to insert in case
                of templated query.

        Returns:
            int: Query execution time in seconds.
        """
        try:
            start = time.monotonic()
            cursor = self.connection.cursor()
            if query_args:
                cursor.execute(query_string % query_args)
            else:
                cursor.execute(query_string)
            runtime = time.monotonic() - start
            cursor.fetchall()
            cursor.close()
        except Exception:
            # Main exception reason seem to be time-outs, just retry.
            self.logger.info("Failed to execute query = {} with args {}, reconnecting..".format(
                query_string, query_args))
            # Wait before reconnecting.
            for i in range(5):
                time.sleep(60)
                try:
                    self.logger.info("Attempting reconnect: {}".format(i))
                    self.connect()
                    break
                except Exception:
                    self.logger.info("Failed reconnection attempt: {}".format(i))
            cursor = self.connection.cursor()
            start = time.monotonic()
            if query_args:
                cursor.execute(query_string % query_args)
            else:
                cursor.execute(query_string)
            runtime = time.monotonic() - start
            cursor.fetchall()
            cursor.close()

        return runtime

    def system_status(self, size_in_gb=True):
        # Important:
        # The Inno-DB storage engine has a bug which causes index size not to be calculated in the statistics
        # table automatically. The update can be triggered by calling 'analyze table t''
        # Hence call this on every table to ensure correct index sizes.
        try:
            cursor = self.connection.cursor()
            for table in self.all_tables:
                db_table = "{}.{}".format(self.db, table)
                cursor.execute("""ANALYZE TABLE %s;""" % db_table)

            # Command below reates outputs: table name, size of data, size of index, total size, e.g.
            # +------------+----------------+-----------------+---------------+
            # | table_name | data_length_mb | index_length_mb | total_size_mb |
            # +------------+----------------+-----------------+---------------+
            # | LINEITEM   | 826.00      | 199.00        | 1025.00     |
            cursor.execute("""
                SELECT table_name,
                round( data_length / ( 1024 *1024 ) , 2 )  AS 'data_length_mb',
                round( index_length / ( 1024 *1024 ) , 2 )  AS 'index_length_mb',
                round( round( data_length + index_length ) / ( 1024 *1024 ) , 2 ) AS 'total_size_mb'
                FROM information_schema.tables
                WHERE table_schema =%s
                ORDER BY data_length desc;""", (self.db, )
            )

            ret = cursor.fetchall()
            cursor.close()
        except Exception:
            self.logger.info("Failed to execute system status, reconnecting")
            # Wait before reconnecting.
            for i in range(5):
                time.sleep(60)
                try:
                    self.logger.info("Attempting reconnect: {}".format(i))
                    self.connect()
                    break
                except Exception:
                    self.logger.info("Failed reconnection attempt: {}".format(i))
            cursor = self.connection.cursor()
            for table in self.all_tables:
                db_table = "{}.{}".format(self.db, table)
                cursor.execute("""ANALYZE TABLE %s;""" % db_table)
            cursor.execute("""
                            SELECT table_name,
                            round( data_length / ( 1024 *1024 ) , 2 )  AS 'data_length_mb',
                            round( index_length / ( 1024 *1024 ) , 2 )  AS 'index_length_mb',
                            round( round( data_length + index_length ) / ( 1024 *1024 ) , 2 ) AS 'total_size_mb'
                            FROM information_schema.tables
                            WHERE table_schema =%s
                            ORDER BY data_length desc;""", (self.db,)
                           )
            ret = cursor.fetchall()
            cursor.close()

        index_size_in_mb = 0.0
        for row in ret:
            index_size_in_mb += float(row[2])

        if size_in_gb:
            return index_size_in_mb / 1024.0, len(self.index_set), self.all_indices
        else:
            return index_size_in_mb, len(self.index_set), self.all_indices

    def is_noop(self, action):
        return action == []

    def reset(self):
        # Manually fetch all indices.
        lift_indices = []
        for table in self.all_tables:
            table_indices = self.show_index_from_table(table)
            for index_tuple in table_indices:
                name = index_tuple[2]
                # Crude way of identifying indices created by LIFT.
                if "_1" in name:
                    # (table, index name)
                    lift_indices.append((index_tuple[0], name))

        cursor = self.connection.cursor()
        for index_tuple in lift_indices:
            table = index_tuple[0]
            index_name = index_tuple[1]
            if index_name not in self.index_set:
                self.logger.info("Removing index {} (table {}) which was not created in this session.".format(
                    index_name, table
                ))
            try:
                cursor.execute("""ALTER TABLE %s DROP INDEX %s""" % (table, index_name))
            except Exception:
                self.logger.debug("Index {} did not exist, could not remove.".format(index_name))

        cursor.close()
        self.index_set.clear()
        self.all_indices = {}

    def drop_index(self, index_spec):
        """
        Utility method to drop specific index for testing without requiring to create it first (as reset
        removes only what was created).

        Args:
            index_spec (dict): Index to drop.
        """

        index_name = self.get_index_name(index_spec["index"])
        table = index_spec["table"]
        try:
            cursor = self.connection.cursor()
            cursor.execute("""ALTER TABLE %s DROP INDEX %s""" % (table, index_name))
            cursor.close()
            self.logger.info("Dropped index = ", index_name)
        except Exception:
            # N.b.: Creating a multi-column index actually creates multiple entries in the
            # index table, but it can only be removed once.
            self.logger.debug("Index did not exist, did not drop index = ", index_name)

    def explain_query(self, query_string, query_args=None):
        """
        Runs EXPLAIN on a query and returns explain results.
        Args:
            query_string (str): Fully formed SQL string.
            query_args(tuple): Optional tuple of string arguments to insert in case
                of templated query.
        """
        query_string = "EXPLAIN " + query_string
        cursor = self.connection.cursor()
        if query_args:
            cursor.execute(query_string % query_args)
        else:
            cursor.execute(query_string)
        return cursor.fetchall()

    def show_index_from_table(self, table):
        """
        Executes the show index command.

        Args:
            table (str):

        Returns:
            any: Result of show index command.
        """
        query_string = """SHOW INDEX FROM {}""".format(table)
        cursor = self.connection.cursor()
        cursor.execute(query_string)
        return cursor.fetchall()

    @staticmethod
    def get_index_name(index):
        """
        Generates a name from an index consisting of one or many columns by concatenating
        column names with underscores and 1 (e.g. index = ["P_BRAND"] -> name = "P_BRAND_1").

        Args:
            index (list): List of columns to index.

        Returns:
            str: Name string.
        """
        return "_".join(["{}_1".format(column) for column in index])

    @staticmethod
    def generate_tpc_h_table_data(path, scale_factor):
        """
        Generates TPC .tbl data with a given load factor. Cleans up any existing .tbl data
        in the dbgen path.

        Args:
            path (str): Path to TPC-H install.
            scale_factor (Union[float, int]): Scale factor for TPC-H (Corresponds to generated data size).
        """
        cmd = "cd {} && rm -rf *.tbl && ./dbgen -s {}".format(path, float(scale_factor))
        local_exec(cmd)

    def generate_tpc_h_schema(self):
        """
        Creates TPC-H tables in MySQL server.
        """
        cmd = "mysql -u {} < /local/scratch/mks40/lift/lift/case_studies/mysql/scripts/create_table.sql".\
            format(self.user)
        local_exec(cmd)

    def load_tpc_h_data(self, path):
        """
        Loads TPC-H tables into db and makes the required schema modifications.

        Args:
            path (str): Path to TPC-H .tbl data directory.
        """
        cmd = "cd {} && mysql -u {}  --local-infile " \
              "< /local/scratch/mks40/lift/lift/case_studies/mysql/scripts/load_tbl_data.sql".format(path, self.user)
        local_exec(cmd)

        # N.b. this will take several minutes to run to create all required keys.
        cmd = "mysql -u {} < /local/scratch/mks40/lift/lift/case_studies/mysql/scripts/alter_schema.sql".\
            format(self.user)
        local_exec(cmd)

