#!/usr/bin/python3

"""
    profile script of postgres index construction time for tpch

    super simple, e.g. consider only complete b-trees

"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import csv
import numpy as np
import time, datetime
from tpch import TPCHClient
from tpch_util import schema


def run():

    ##
    # choose indices
    # 
    
    # selected these based on scale 
    # lineitem index is worst-case, sf*6,000,000 records roughly
    # supplier is on smaller side, sf*10,000 records roughly
    rels = ['lineitem', 'partsupp', 'part', 'supplier']

    idxs = []
    for rel in rels:
        attrs = list(np.random.choice(schema[rel], size=1, replace=False))
        idx = '_'.join([rel] + attrs)
        idxs.append((idx, rel, attrs))
        
        attrs = list(np.random.choice(schema[rel], size=2, replace=False))
        idx = '_'.join([rel] + attrs)
        idxs.append((idx, rel, attrs))


    ##
    # create indices for different db sizes
    #

    tpch = TPCHClient()

    sfs = [.1, .2] # TBU
    times = []

    for sf in sfs:
    
        print('#################### sf = {} ####################'.format(sf)) 
        tpch.repopulate(sf)

        for idx in idxs:

            idx_id, rel, attrs = idx 

            # not sure how to capture this exactly?
            # client is in auto-commit and blocks
            # this seems sufficient... maybe look at logs?
            # https://dba.stackexchange.com/questions/11329/monitoring-progress-of-index-construction-in-postgresql
            tic=time.time()
            tpch.set_index(idx_id, rel, attrs)
            toc=time.time()
            print('built {} in {} seconds'.format(idx_id, round(toc-tic, 2)))
            
            times.append((idx_id, sf, round(toc-tic)))
            
    tpch.close()

    ##
    # write
    #
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%m-%d %H-%M-%S") #ha
    saveas = str(min(sfs)) + '-' + str(max(sfs)) + ' ' + timestamp + '.csv' 
    # np.savetxt(saveas, times, delimiter=',', fmt="%s") # %s truncates 
    with open(saveas,'w') as f:
        writer = csv.writer(f, delimiter=',')
        for r in times:
            writer.writerow(r)

    return times

def viz(times):

    # bar chart data
    times = np.array(times)
    bar_labels = times[:,0]
    bar_loc = range(times.shape[0])
    vals = times[:,2].astype(np.float16)
    colors = np.array([(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44)]) / 255.    
    _, colors_idx = np.unique(times[:,1], return_inverse=True)
    colors = colors[colors_idx]

    # bar chart
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bars = ax.bar(bar_loc, vals, color=colors)

    # labels, legend    
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)    

    plt.xticks(bar_loc, bar_labels, rotation=90, fontsize=8)
    _, unique_idx = np.unique(times[:,1], return_index=True)
    ax.legend(np.array(bars)[unique_idx], times[:,1][unique_idx], fontsize=8)

    ax.set_title("Index creation times in seconds \nfor different database scale factors", fontsize = 8)
    plt.tight_layout()
    fig.savefig('res.png')


if __name__ == "__main__":
    
    data = run()
    viz(data)