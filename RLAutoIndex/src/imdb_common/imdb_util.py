from cgi import print_directory
import numpy as np
import string
import csv
import random
import os


imdb_tables = ["title"]

from collections import OrderedDict

# TODO 3.5 does not give any guarantee about dict keys
imdb_table_columns_ = OrderedDict()
imdb_table_columns_['title'] = ['id','title','imdb_index','kind_id','production_year','imdb_id','phonetic_code','episode_of_id','season_nr','episode_nr','series_years'], 



#  table -> columns -> types, values
imdb_table_columns = {
    "title": {
        "id":[int,"fixed_range",[100002,4736508]],
        "title":[str,"sample_fn_k"],
        "imdb_index":[str,"sample_fn"],
        "kind_id":[int,"sample_fn"],
        "production_year":[str,"text"],
        "imdb_id":[str,"look_up"],
        "phonetic_code":[str,"sample_fn_k"],
        "episode_of_id":[int, "sample_fn"],
        "season_nr":[int,"sample_fn"],
        "episode_nr":[int,"sample_fn"],
        "series_years":[str,"sample_fn_k"],
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

production_year = ['1933', '1908', '1945', '2015', '2000', '1923', '1936', '1957', '1994', 
'2018', '1899', '1891', '1975', '1968', '2010', '1972', '1948', '1913', '1900', '1912', 
'1989', '2007', '1930', '1895', '1976', '1940', '1909', '1938', '2004', '1954', '1890', 
'1969', '1998', '2023', '1889', '1926', '1981', '1959', '1986', '1997', '1977', '1963', 
'1956', '1922', '1951', '2026', '1983', '1887', '1932', '1917', '1992', '2011', '2017', 
'1903', '1955', '1874', '1979', '1924', '1947', '1901', '1904', '1925', '1939', '1960', 
'1984', '1966', '1920', '1980', '2013', '1964', '1991', '1961', '1962', '1978', '2020', 
'1898', '1982', '2012', '1995', '2021', '2001', '1910', '1973', '1958', '2002', '1950', 
'1952', '1897', '1929', '1970', '1916', '2025', '1888', '1971', '1907', '1896', '1941', 
'1911', '1937', '2006', '1942', '1987', '1905', '1919', '1985', '1996', '1931', '2014', 
'2115', '2008', '1965', '2005', '1894', '1893', '1993', '1878', '1967', '1902', '2003', 
'1974', '1944', '1928', '1918', '1915', '2009', '1946', '2022', '1914', '1883', '1999', 
'1934', '2016', '1927', '1949', '1906', '1892', '2019', '1921', '1988', '1953', '2024', '1990', '1943', '1935']

TEXT_WORDS = ['sleep', 'haggle', 'affix', 'nod', 'solve', 'hinder', 'eat', 'poach', 'snooze', 'play',
              'furious', 'quick', 'ruthless', 'daring', 'enticing', 'final', 'silent', 'furious', 'quick',
              'ruthless', 'daring', 'enticing', 'final', 'silent', 'about', 'after', 'among', 'before', 'besides',
              'despite', 'from', 'into', 'outside', 'through', 'under', 'without']

IMDB_ID = [""]
#
# table of hardcoded values
# 

imdb_string_values = {
    "imdb_id":IMDB_ID 
}


#
# samplers
#

def sample_text():
    # Text generation is tricky because comments are just random sub-strings:
    # SELECT L_COMMENT FROM LINEITEM ORDER BY RAND() LIMIT 1; ->
    # "ording to the slyly special package", "oxes cajole fluffil"
    # TPC-H specifies a full grammar for sentences, but then randomly cut sub-strings are selected/
    # We sill simply do %LIKE% queries on individual words:
    # SELECT L_COMMENT FROM LINEITEM WHERE L_COMMENT LIKE '%blithely%' ORDER BY RAND() LIMIT 1;
    return np.random.choice(production_year)



def random_float(low, high):
    return np.random.random() * (high - low) + low



def sample_fnr1(n):
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile)   
        column = [row[1] for row in reader]

    rand_low = random.randint(0,len(column))
    rand_high = rand_low+n
    list =column[rand_low:rand_high]
    k = random.choice(list)
    k = k.replace("'","_")
    for _ in range(10):
        if "_" in k:
            rand_low = random.randint(0,len(column))
            rand_high = rand_low+n
            list =column[rand_low:rand_high]
            k = random.choice(list)
            k = k.replace("'","_")
        else:
            break
    return(k)



def sample_fnr2():
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile) 
        for row in reader:
            row_list = list([row[2]])
            #if list([row[2]]) == ['XVII']:
            #    print([row])

    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile) 
        column = [row[2] for row in reader]
        conditional_col = column.index('XVII')

    #print(conditional_col)

    col = list(set(column))
    return(random.choice(col))

def sample_fnr3():
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile)   
        column = [row[3] for row in reader]

    col = list(set(column))
    return(random.choice(col))


def sample_fnr6(k):
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile)   
        l_list = []
        n_list = []
        l_n_list = []
        for row in reader:
            row_list = list([row[6]])
            #print(row_list )
            #print(row_list)
            if row_list != ['']:
                row_list = list(row_list[0])
                l = row_list.pop(0)
                n = "".join(row_list)
                l_list.append(l)
                n_list.append(n)
        
        for _ in range(k):
            l = random.choice(l_list)
            n = random.choice(n_list)
            ln = (l,n)
            l_n = ''.join(ln)
            l_n_list.append(l_n)

    return random.choice(l_n_list)
    
def sample_fnr7():
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile) 
        column=[]
        for row in reader:
            if list([row[7]])!= ['']: 
                column.append(int(row[7]))
        column.sort()

    col = sorted(set(column))
    return(random.choice(col))

def sample_fnr8():
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile)   
        column=[]
        for row in reader:
            if list([row[8]])!= ['']: 
                column.append(int(row[8]))
        #column.sort()
    col = sorted(set(column))
    return(random.choice(col))

def sample_fnr9():
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile)   
        column=[]
        for row in reader:
            if list([row[9]])!= ['']: 
                column.append(int(row[9]))
        column.sort()
    col = sorted(set(column))
    return random.choice(col)

def sample_fnr10(n):
    str_list = []
    yr_start_list =[]
    yr_end_list = []
    with open('test/title.csv','r') as csvfile:
        reader = csv.reader(csvfile)   
        for row in reader:
            row_list = list([row[10]])
            #print(row_list )
            #print(row_list)
            if row_list != ['']:
                row_list = row_list[0].split('-')
                yr_start = row_list.pop(0)
                yr_end = row_list
                #n = "".join(row_list)
                yr_start_list.append(yr_start)
                if yr_end != []:
                    yr_end_0 = yr_end[0]
                    yr_end_list.append(yr_end_0)
    #print(sorted(set(yr_start_list)))
    #print(sorted(set(yr_end_list)))
   
    str_1 = random.choice(yr_start_list)
    str_2 = random.choice(yr_end_list)
    
    for _ in range(999):
        if str_1 != '????' and str_2 != '????':
            if int(str_1)>int(str_2):
                str_1 = random.choice(yr_start_list)
                str_2 = random.choice(yr_end_list)
            else:
                break
        else:
            str_1 = random.choice(yr_start_list)
            str_2 = random.choice(yr_end_list)

    str = (str_1, str_2)

    str = '-'.join(str)
    str_list.append(str)
    return str_list[0]


#
# table of samplers
#

imdb_sample_fns = dict(
    title = sample_fnr1,
    imdb_index = sample_fnr2,
    kind_id = sample_fnr3,
    phonetic_code = sample_fnr6,
    episode_of_id = sample_fnr7,
    season_nr= sample_fnr8,
    episode_nr = sample_fnr9,
    series_years = sample_fnr10,
)
