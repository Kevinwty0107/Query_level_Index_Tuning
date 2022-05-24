import numpy as np
import string

"""
Comments:
  - boilerplated from lift.case_studies
  - see section 4.2 in http://www.tpc.org/tpc_documents_current_versions/pdf/tpc-h_v2.17.3.pdf
  
  - TODO step through this, cleanup as you can

"""

tpch_tables = ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

from collections import OrderedDict

# TODO 3.5 does not give any guarantee about dict keys
tpch_table_columns_ = OrderedDict()
tpch_table_columns_['lineitem'] = ['L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY', 'L_LINENUMBER', 'L_QUANTITY', 'L_EXTENDEDPRICE', 'L_DISCOUNT', 'L_TAX', 'L_RETURNFLAG', 'L_LINESTATUS', 'L_SHIPDATE', 'L_COMMITDATE', 'L_RECEIPTDATE', 'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_COMMENT'], 
tpch_table_columns_['partsupp'] = ['PS_PARTKEY', 'PS_SUPPKEY', 'PS_AVAILQTY', 'PS_SUPPLYCOST', 'PS_COMMENT'], 
tpch_table_columns_['region'] = ['R_REGIONKEY', 'R_NAME', 'R_COMMENT'], 
tpch_table_columns_['orders'] = ['O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS', 'O_TOTALPRICE', 'O_ORDERDATE', 'O_ORDERPRIORITY', 'O_CLERK', 'O_SHIPPRIORITY', 'O_COMMENT'], 
tpch_table_columns_['nation'] = ['N_NATIONKEY', 'N_NAME', 'N_REGIONKEY', 'N_COMMENT'], 
tpch_table_columns_['part'] = ['P_PARTKEY', 'P_NAME', 'P_MFGR', 'P_BRAND', 'P_TYPE', 'P_SIZE', 'P_CONTAINER', 'P_RETAILPRICE', 'P_COMMENT'], 
tpch_table_columns_['supplier'] = ['S_SUPPKEY', 'S_NAME', 'S_ADDRESS', 'S_NATIONKEY', 'S_PHONE', 'S_ACCTBAL', 'S_COMMENT'], 
tpch_table_columns_['customer'] = ['C_CUSTKEY', 'C_NAME', 'C_ADDRESS', 'C_NATIONKEY', 'C_PHONE', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_COMMENT']


#  table -> columns -> types, values
tpch_table_columns = {
    "nation": {
        "N_NATIONKEY": [int, "fixed_range", [0, 25]],
        "N_NAME": [str, "lookup"],
        "N_REGIONKEY": [int, "fixed_range", [0, 6]],
        "N_COMMENT": [str, "text", [31, 114]]
    },
    "region": {
        "R_REGIONKEY": [int, "fixed_range", [0, 5]],
        "R_NAME": [str, "lookup"],
        "R_COMMENT": [str, "text", [31, 115]]
    },
    "part": {
        "P_PARTKEY": [int, "scaled_range", [0, 200000]],
        "P_NAME": [str, "sample_fn"],
        "P_MFGR": [str, "sample_fn"],
        "P_BRAND": [str, "sample_fn"],
        "P_TYPE": [str, "sample_fn"],
        "P_SIZE": [int, "fixed_range", [0, 51]],
        "P_CONTAINER": [str, "sample_fn"],
        "P_RETAILPRICE": [float, "fixed_range", [1000.0, 2000.0]],
        "P_COMMENT": [str, "text", [5, 23]]
    },
    "supplier": {
        "S_SUPPKEY": [int, "scaled_range", [0, 10000]],
        "S_NAME": [str, "scaled_sample_fn"],
        "S_ADDRESS": [str, "sample_fn"],
        "S_PHONE": [str, "sample_fn"],
        "S_ACCTBAL": [float, "sample_fn"],
        "S_COMMENT": [str, "text", [5, 23]]
    },
    "partsupp": {
        "PS_PARTKEY": [int, "scaled_range", [0, 10000]],
        "PS_SUPPKEY": [int, "scaled_range", [0, 10000]],
        "PS_AVAILQTY": [int, "fixed_range", [1, 10000]],
        "PS_SUPPLYCOST": [float, "fixed_range", [1.0, 1000.0]],
        "PS_COMMENT": [str, "text", [49, 199]]
    },
    "customer": {
        "C_CUSTKEY": [int, "scaled_range", [0, 150000]],
        "C_NAME": [str, "scaled_sample_fn"],
        "C_ADDRESS": [str, "sample_fn"],
        "C_NATIONKEY": [int, "fixed_range", [0, 25]],
        "C_PHONE": [str, "sample_fn"],
        "C_ACCTBAL": [float, "sample_fn"],
        "C_MKTSEGMENT": [str, "lookup"],
        "C_COMMENT": [str, "text", [29, 117]]
    },
    "orders": {
        "O_ORDERKEY": [int, "scaled_range", [0, 6000000]],
        "O_CUSTKEY": [int, "scaled_range", [0, 150000]],
        "O_ORDERSTATUS": [str, "lookup"],
        "O_TOTALPRICE": [float, "fixed_range", [100000.0, 200000.0]],
        "O_ORDERDATE": ["date", "sample_fn"],
        "O_ORDERPRIORITY": [str, "lookup"],
        "O_CLERK": [str, "scaled_sample_fn"],
        "O_SHIPPRIORITY": [int, "fixed_range", [0, 1]],
        "O_COMMENT": [str, "text", [19, 79]]
    },
    "lineitem": {
        "L_ORDERKEY": [int, "scaled_range", [0, 6000000]],
        "L_PARTKEY": [int, "scaled_range", [0, 200000]],
        "L_SUPPKEY": [int, "scaled_range", [0, 10000]],
        "L_LINENUMBER": [int, "fixed_range", [0, 8]],
        "L_QUANTITY": [int, "fixed_range", [1, 51]],
        "L_EXTENDEDPRICE": [float, "fixed_range", [1000.0, 150000.0]],
        "L_DISCOUNT": [float, "fixed_range", [0.0, 0.1]],
        "L_TAX": [float, "fixed_range", [0.0, 0.08]],
        "L_RETURNFLAG": [str, "lookup"],
        "L_LINESTATUS": [str, "lookup"],
        "L_SHIPDATE":  ["date", "sample_fn"],
        "L_COMMITDATE":  ["date", "sample_fn"],
        "L_RECEIPTDATE":  ["date", "sample_fn"],
        "L_SHIPINSTRUCT": [str, "lookup"],
        "L_SHIPMODE": [str, "lookup"],
        "L_COMMENT": [str, "text", [10, 44]]
    }
}

column_type_operators = {
    int: ["=", "<", ">"],
    float: ["=", "<", ">"],
    str: ["="],
    "date": ["<", ">"],
    "text": ["LIKE"]
}

#
# hardcoded values
#

N_NAME = ['(0, ALGERIA, 0)', '(1, ARGENTINA, 1)', '(2, BRAZIL, 1)',
          '(3, CANADA, 1)', '(4, EGYPT, 4)', '(5, ETHIOPIA, 0)',
          '(6, FRANCE, 3)', '(7, GERMANY, 3)', '(8, INDIA, 2)', '(9, INDONESIA, 2)',
          '(10,IRAN, 4)', '(11, IRAQ, 4)', '(12, JAPAN, 2)', '(13, JORDAN, 4)',
          '(14, KENYA, 0)', '(15, MOROCCO, 0)', '(16, MOZAMBIQUE, 0)', '(17, PERU, 1)',
          '(18, CHINA, 2)', '(19, ROMANIA, 3)', '(20, SAUDI ARABIA, 4)', '(21, VIETNAM, 2)',
          '(22, RUSSIA, 3)', '(23, UNITED KINGDOM, 3)', '(24, UNITED STATES, 1)']
R_NAME = ['AFRICA', 'AMERICA', 'ASIA', 'EUROPE', 'MIDDLE EAST']

P_NAME = ["almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched", "blue",
          "blush", "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral", "cornflower",
          "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick", "floral", "forest",
          "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew", "hot", "indian", "ivory",
          "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen", "magenta", "maroon", "medium",
          "metallic", "midnight", "mint", "misty", "moccasin", "navajo", "navy", "olive", "orange", "orchid",
          "pale", "papaya", "peach", "peru", "pink", "plum", "powder", "puff", "purple", "red", "rose", "rosy",
          "royal", "saddle", "salmon", "sandy", "seashell", "sienna", "sky", "slate", "smoke", "snow", "spring",
          "steel", "tan", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "yellow"]

# 4.2.2.13
TYPES_SIZE = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
TYPES_ADJ = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
TYPES_MATERIAL = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
CONTAINER_SIZE = ["SM", "LG", "MED", "JUMBO", "WRAP"]
CONTAINER_PACKAGE = ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"]

# 4.2.2.14, selected some
TEXT_WORDS = ['sleep', 'haggle', 'affix', 'nod', 'solve', 'hinder', 'eat', 'poach', 'snooze', 'play',
              'furious', 'quick', 'ruthless', 'daring', 'enticing', 'final', 'silent', 'furious', 'quick',
              'ruthless', 'daring', 'enticing', 'final', 'silent', 'about', 'after', 'among', 'before', 'besides',
              'despite', 'from', 'into', 'outside', 'through', 'under', 'without']

SEGMENTS = ["AUTOMOBILE", "BUILDING", "HOUSEHOLD", "FURNITURE", "MACHINERY"]
O_ORDERSTATUS = ["F", "O", "P"]
START_DATE = [1992, 1, 1]
END_DATE = [1998, 12, 31]

O_ORDERPRIORITY = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
L_RETURNFLAG = ["R", "A", "N"]
L_LINESTATUS = ["O", "F"]
L_SHIPINSTRUCT = ["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"]
L_SHIPMODE = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]

#
# table of hardcoded values
# 

tpch_string_values = {
    "N_NAME": N_NAME,
    "R_NAME": R_NAME,
    "C_MKTSEGMENT": SEGMENTS,
    "O_ORDERSTATUS": O_ORDERSTATUS,
    "L_RETURNFLAG": L_RETURNFLAG,
    "L_LINESTATUS": L_LINESTATUS,
    "O_ORDERPRIORITY": O_ORDERPRIORITY,
    "L_SHIPINSTRUCT": L_SHIPINSTRUCT,
    "L_SHIPMODE": L_SHIPMODE
}


#
# samplers
#

def sample_p_name():
    samples = np.random.choice(P_NAME, size=5, replace=False)
    return " ".join(samples)

def sample_p_mfgr(): return "Manufacturer#{}".format(np.random.randint(0, 6))

def sample_p_brand(): return "Brand#{}{}".format(np.random.randint(0, 6), (np.random.randint(0, 6)))

def sample_p_type():
    return "{} {} {}".format(np.random.choice(TYPES_SIZE),
                             np.random.choice(TYPES_ADJ), np.random.choice(TYPES_MATERIAL))

def sample_p_container():
    return "{} {}".format(np.random.choice(CONTAINER_SIZE), np.random.choice(CONTAINER_PACKAGE))

def sample_text():
    # Text generation is tricky because comments are just random sub-strings:
    # SELECT L_COMMENT FROM LINEITEM ORDER BY RAND() LIMIT 1; ->
    # "ording to the slyly special package", "oxes cajole fluffil"
    # TPC-H specifies a full grammar for sentences, but then randomly cut sub-strings are selected/
    # We sill simply do %LIKE% queries on individual words:
    # SELECT L_COMMENT FROM LINEITEM WHERE L_COMMENT LIKE '%blithely%' ORDER BY RAND() LIMIT 1;
    return np.random.choice(TEXT_WORDS)

def sample_s_name(scale_factor):
    # e.g. Supplier#000000088
    max_val = scale_factor * 10000
    sample = np.random.randint(0, max_val)
    leading_zeros = 9 - len(str(sample))
    prefix = ['0'] * leading_zeros
    # prefix = 000000, sample = 100
    return "Supplier#{}{}".format("".join(prefix), sample)

chars = [c for c in string.ascii_letters]

def random_string(min_length, max_length):
    num_samples = np.random.randint(low=min_length, high=max_length)
    return ''.join(np.random.choice(chars, size=num_samples))

def sample_s_address():
    return random_string(10, 40)

def phone_number():
    # country_code, "-", local_number1, "-", local_number2, "-", local_number3
    country_code = np.random.randint(10, 34)
    local_1 = np.random.randint(100, 1000)
    local_2 = np.random.randint(100, 1000)
    local_3 = np.random.randint(1000, 10000)
    return "{}-{}-{}-{}".format(country_code, local_1, local_2, local_3)

def random_float(low, high):
    return np.random.random() * (high - low) + low

def sample_s_acct_balance():
    return random_float(-999.99, 9999.99)

def sample_c_name(scale_factor):
    # e.g. Customer#000000088
    max_val = scale_factor * 10000
    sample = np.random.randint(0, max_val)
    leading_zeros = 9 - len(str(sample))
    prefix = ['0'] * leading_zeros
    # prefix = 000000, sample = 100
    return "Customer#{}{}".format("".join(prefix), sample)

days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

def random_date_in_range():
    # Very low tech solution to not bother with date formats..
    year = START_DATE[0] + np.random.randint(0, 7)
    month = np.random.randint(1, 13)
    day = np.random.randint(1, days_in_month[month] + 1)
    month = "0{}".format(month) if month < 10 else month
    day = "0{}".format(day) if day < 10 else day
    return "{}-{}-{}".format(year, month, day)

def sample_o_clerk(scale_factor):
    # e.g.Clerk#000000088
    max_val = scale_factor * 10000
    sample = np.random.randint(0, max_val)
    leading_zeros = 9 - len(str(sample))
    prefix = ['0'] * leading_zeros
    # prefix = 000000, sample = 100
    return "Clerk#{}{}".format("".join(prefix), sample)

#
# table of samplers
#

tpch_sample_fns = dict(
    P_NAME=sample_p_name,
    P_MFGR=sample_p_mfgr,
    P_BRAND=sample_p_brand,
    P_TYPE=sample_p_type,
    P_CONTAINER=sample_p_container,
    S_NAME=sample_s_name,
    S_ADDRESS=sample_s_address,
    S_PHONE=phone_number,
    S_ACCTBAL=sample_s_acct_balance,
    C_NAME=sample_c_name,
    C_ADDRESS=sample_s_address,
    C_PHONE=phone_number,
    C_ACCTBAL=sample_s_acct_balance,
    O_ORDERDATE=random_date_in_range,
    O_CLERK=sample_o_clerk,
    # Just random dates.
    L_SHIPDATE=random_date_in_range,
    L_COMMITDATE=random_date_in_range,
    L_RECEIPTDATE=random_date_in_range
)
