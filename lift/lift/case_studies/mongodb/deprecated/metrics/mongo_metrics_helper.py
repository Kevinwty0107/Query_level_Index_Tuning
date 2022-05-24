from lift.case_studies.mongodb.deprecated.metrics.metrics_helper import MetricsHelper
import numpy as np
import csv

from lift.util.parsing_util import serialize_ops


class MongoMetricsHelper(MetricsHelper):
    """
    Records metrics for MongoDB query execution.
    """

    def __init__(self):

        # Things we want to record
        super(MongoMetricsHelper, self).__init__()
        self.metrics = dict(
            action_info=list(),
            index_sizes=list(),
            index_creation_info=list(),
            rewards=list(),
            runtimes=list(),
            batch_rewards=list(),
            pretrain_info=list(),
            serialized_rows=list(),
            raw_queries=list()
        )

    def serialize_observations(self, states):
        for observation in states:
            # Convert op fields to string
            meta_data = observation.get_meta_data()
            if meta_data['index_name'] is None:
                index_name = 'none'
            else:
                index_name = meta_data['index_name']

            serialized_fields = serialize_ops(meta_data['op_field_list'])
            metrics = dict(
                raw_queries=str(observation.get_query()),
                serialized_rows=(
                    str(meta_data['sort_info']),
                    serialized_fields,
                    observation.get_runtime(),
                    meta_data['index_size'],
                    index_name
                )
            )
            self.record_result(metrics)

    def record_observations(self, observations, kwargs):
        self.record_result(
            dict(
                rewards=[parsed_entry.get_meta_data()['reward'] for parsed_entry in observations],
                index_sizes=[parsed_entry.get_meta_data()['index_size'] for parsed_entry in observations],
                runtimes=[parsed_entry.get_meta_data()['runtime'] for parsed_entry in observations],
                batch_rewards=kwargs['batch_reward']
            )
        )

    def export_results(self, path, kwargs):
        serial_path = kwargs['serial_path']
        serialize = kwargs['serialize']
        # Actions is list (one entry per interval) of list (one entry per batch item) of lists
        sizes = np.array(self.metrics['index_sizes'])
        runtimes = np.array(self.metrics['runtimes'])
        rewards = np.array(self.metrics['rewards'])
        pretraining = np.array(self.metrics['pretrain_info'])
        batch_rewards = np.array(self.metrics['batch_rewards'])

        action_path = path + '/actions.txt'
        action_batch_path = path + '/batches.txt'
        size_path = path + '/sizes.txt'
        runtime_path = path + '/runtimes.txt'
        rewards_path = path + '/rewards.txt'
        pretraining_path = path + '/pretrain_info.txt'
        index_creations_path = path + '/index_creations_info.txt'
        batch_rewards_path = path + '/batch_rewards.txt'

        np.savetxt(size_path, sizes, delimiter=',')
        np.savetxt(runtime_path, runtimes, delimiter=',')
        np.savetxt(rewards_path, rewards, delimiter=',')
        np.savetxt(pretraining_path, pretraining, delimiter=',')
        np.savetxt(batch_rewards_path, batch_rewards, delimiter=',')

        if serialize:
            raw_query_path = serial_path + '_queries.csv'
            data_path = serial_path + '_data.csv'
            with open(data_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in self.metrics['serialized_rows']:
                    writer.writerow(row)

            with open(raw_query_path, 'a', newline='') as f:
                for row in self.metrics['raw_queries']:
                    f.write(row + '\n')

        # Export all actions done in a csv format
        with open(action_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for batch in self.metrics['action_info']:
                # Each list entry in batch
                for full_action in batch:
                    writer.writerow(full_action)

        # Export batch-wise info
        with open(action_batch_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for batch in self.metrics['action_info']:
                batch_info = [len(batch)]
                noops = 0
                indices = 0

                for full_action in batch:
                    if full_action == [0]:
                        noops += 1
                    else:
                        indices += 1

                batch_info.append(noops)
                batch_info.append(indices)
                writer.writerow(batch_info)

        # Export index creation info
        with open(index_creations_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for index_info in self.metrics['index_creation_info']:
                writer.writerow(index_info)
