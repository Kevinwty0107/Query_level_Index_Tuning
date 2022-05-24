import unittest

from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
from lift.case_studies.mysql.tpch_util import tpch_tables


class TestMySQLSystemEnvironment(unittest.TestCase):

    """
    Tests index manipulation and query execution in MySQL.

    Use in conjunction with CLI to verify correct database state.


    """
    def test_index_creation(self):
        """
        Tests act() method.

        python -m pytest -s lift/case_studies/mysql/tests/test_mysql_system_model.py::TestMySQLSystemEnvironment::test_index_creation
        """
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        # Test single index.
        index = dict(
            index=["P_BRAND"],
            table="PART"
        )

        # Drop this index so test is idempotent.
        system.drop_index(index)

        status = system.system_status()

        # Create simple index.
        print("Size before {} mb".format(status))
        system.act(index)

        status = system.system_status()
        print("Size after {} mb".format(status))

        # Test compound index
        compound_index = dict(
            index=["P_TYPE", "P_SIZE"],
            table="PART"
        )
        system.drop_index(compound_index)
        status = system.system_status()

        # Create simple index.
        print("Size before {} GB".format(status))
        system.act(compound_index)

        status = system.system_status()
        print("Size after {} GB".format(status))

        # Comment out below, log into CLI, check index exists via
        # SHOW INDEX FROM PART;
        system.drop_index(index)
        system.drop_index(compound_index)

    def test_index_prefixing(self):
        """
        Tests how MySQL handles prefixes and duplicate indices. If index A-B-C, what happens if we try to create
         index A or index B separately?

        python -m pytest -s lift/case_studies/mysql/tests/test_mysql_system_model.py::TestMySQLSystemEnvironment::test_index_prefixing
        """
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)
        system.reset()
        # Test single index.
        index = dict(
            index=["P_BRAND", "P_TYPE", "P_SIZE"],
            table="PART"
        )
        system.act(index)
        indices = system.show_index_from_table("PART")
        lift_indices = []
        for index_tuple in indices:
            name = index_tuple[2]
            # Crude way of identifying indices created by LIFT.
            if "_1" in name:
                lift_indices.append(name)
        print("Created compound index: ", index["index"])
        print("Num all indices = ", len(indices))
        print("Num lift indices =", len(lift_indices))
        print("Lift indices = ", lift_indices)
        status = system.system_status()
        print("Size after {} GB".format(status))

        index = dict(
            index=["P_BRAND"],
            table="PART"
        )
        print("Now creating first prefix:", index["index"])
        system.act(index)
        indices = system.show_index_from_table("PART")
        lift_indices = []
        for index_tuple in indices:
            name = index_tuple[2]
            # Crude way of identifying indices created by LIFT.
            if "_1" in name:
                lift_indices.append(name)
        print("Created additional prefix index: ",  index["index"])
        print("Num all indices = ", len(indices))
        print("Num lift indices =", len(lift_indices))
        print("Lift indices = ", lift_indices)
        status = system.system_status()
        print("Size after {} GB".format(status))

        index = dict(
            index=["P_TYPE"],
            table="PART"
        )
        print("Now creating index from middle column:", index["index"])
        system.act(index)
        indices = system.show_index_from_table("PART")
        lift_indices = []
        for index_tuple in indices:
            name = index_tuple[2]
            # Crude way of identifying indices created by LIFT.
            if "_1" in name:
                lift_indices.append(name)
        print("Created additional index index: ", index["index"])
        print("Num all indices = ", len(indices))
        print("Num lift indices =", len(lift_indices))
        print("Lift indices = ", lift_indices)
        status = system.system_status()
        print("Size after {} GB".format(status))
        system.reset()

    def test_status(self):
        # python -m pytest -s lift/case_studies/mysql/tests/test_mysql_system_model.py::TestMySQLSystemEnvironment::test_status
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)
        print("Index size =", system.system_status())

    def test_show_index(self):
        """
        Test show index -> try to fetch all indices for a safer reset.

        # python -m pytest -s lift/case_studies/mysql/tests/test_mysql_system_model.py::TestMySQLSystemEnvironment::test_show_index
        """
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        # Debug - find out structure of the result:
        indices = system.show_index_from_table("PART")
        lift_indices = []
        for index_tuple in indices:
            name = index_tuple[2]
            # Crude way of identifying indices created by LIFT.
            if "_1" in name:
                lift_indices.append(name)
        print("Num all indices = ", len(indices))
        print("Num lift indices =", len(lift_indices))
        print("Lift indices = ", lift_indices)

        system.reset()
        indices = system.show_index_from_table("PART")
        print("Num indices after idempotent reset = ", len(indices))

    def test_query_execution(self):
        # python -m pytest -s lift/case_studies/mysql/tests/test_mysql_system_model.py::TestMySQLSystemEnvironment::test_query_execution
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        tpch_10_query = """select c_custkey,
            c_name,
            sum(l_extendedprice * (1 - l_discount)) as revenue,
            c_acctbal,
            n_name,
            c_address,
            c_phone,
            c_comment
        from
            CUSTOMER,
            ORDERS,
            LINEITEM,
            NATION
        where
            c_custkey = o_custkey
            and l_orderkey = o_orderkey
            and o_orderdate >= date '1993-08-01'
            and o_orderdate < date '1993-08-01' + interval '3' month
            and l_returnflag = 'R'
            and c_nationkey = n_nationkey
        group by
            c_custkey,
            c_name,
            c_acctbal,
            c_phone,
            n_name,
            c_address,
            c_comment
        order by
            revenue desc
        limit 20;
        """

        # Check warmup effects.
        for _ in range(10):
            print("Executing tpc-h query #10, runtime is {} s ".format(system.execute(tpch_10_query)))
