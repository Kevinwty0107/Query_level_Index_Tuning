import csv

# Exporting/eval for baseline comparisons.


def export(result_dir, label, runtimes, index_size, num_indices, final_index, queries=None, indices=None):
    path = result_dir + '/' + label

    # Export runtime tuples.
    with open(path + '_runtimes.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for row in runtimes:
            writer.writerow(row)

    # Export queries with their respective indices and runtimes for easier manual analysis.
    with open(path + '_query_index_runtime.csv', 'a', newline='') as f:
        writer = csv.writer(f, delimiter='-')
        for i in range(len(runtimes)):
            runtime_tuple = runtimes[i]
            query = queries[i]
            if indices:
                index = indices[i]
            else:
                index = 'None'
            writer.writerow([runtime_tuple[0], runtime_tuple[1], query, index])

    # Export index size.
    with open(path + '_index_size.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([index_size])
        writer.writerow([num_indices])

    final_index_path = result_dir + '/final_index/' + label + '_final_index.json'
    with open(final_index_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for index_key, value in final_index.items():
            writer.writerow([str(index_key), str(value)])
