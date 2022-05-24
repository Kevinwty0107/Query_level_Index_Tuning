from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import csv
import heapq
import itertools

def test_action_format():
    action_info = []

    # append no-op
    # append compound index
    batch = [[0], [1, 2]]
    action_info.append(batch)

    other_batch = [[0], [1]]
    action_info.append(other_batch)

    with open('test.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for batch in action_info:

            for full_action in batch:
                writer.writerow(full_action)
                print(full_action)
    with open('batch.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        for batch in action_info:
            batch_info = [len(batch)]
            noops = 0
            indices = 0
            for full_action in batch:
                if full_action == [0]:
                    noops += 1
                else:
                    indices += 1
                print(full_action)
            batch_info.append(noops)
            batch_info.append(indices)
            writer.writerow(batch_info)
            # Now list of lists with actions

def test_index_name():
    name = "user_role_1_location_1"

    fields = ['location', 'user_role', 'status']
    count = 0
    for field in fields:
        # We find the index of fields which are in the name of the index but not in right order
        index = name.find(field)
        if index >= 0:
            # Field found in index
            print(index)
            add_task(field, index)
            count += 1

    index_columns = [pop_task() for _ in range(count)]
    print(index_columns)




pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task
counter = itertools.count()     # unique sequence count

def add_task(task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heapq.heappush(pq, entry)

def remove_task(task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while pq:
        priority, count, task = heapq.heappop(pq)
        if task is not REMOVED:
            del entry_finder[task]
            return task
    raise KeyError('pop from an empty priority queue')