import ast
import os
import csv
import json
import numpy as np
from lift.case_studies.heron.heron_agent_model_generator import \
    HeronAgentModelGenerator
from lift.case_studies.heron.non_zero_scaler import NonZeroScaler
from lift.model.state import State


class HeronFullStateModelGenerator(HeronAgentModelGenerator):

    def __init__(self, constant_state, latency_normaliser,
                 throughput_normaliser, reward_generator, experiment_config):
        super(HeronFullStateModelGenerator, self).__init__(constant_state,
                                                           latency_normaliser, throughput_normaliser, reward_generator,
                                                           experiment_config)
        self.max_instances = self.experiment_config['max_instances']
        self.bolts = self.experiment_config['bolts']
        load_from_json = self.experiment_config['model_generator_args'] \
            ['load_from_json']

        # need a separate scaler for every metric 
        self.scalers = dict()
        for metric in self.metric_to_index.keys():
            self.scalers[metric] = NonZeroScaler()
        if load_from_json:
            json_file = os.path.join(
                self.experiment_config['model_generator_args']['json_dir'],
                self.experiment_config['model_generator_args']['json_file']
            )
            metrics_to_scaler_params = self._load_dict_from_json(json_file)
            self._init_scaler_with_dict(metrics_to_scaler_params)
        else:
            message_file = os.path.join(self.experiment_config['demo_dir'],
                                        self.experiment_config['demo_file'])
            self.logger.info("Loading messages from path: {}".format(
                message_file
            ))
            messages = self._load_messages_from_file(message_file)
            self._init_scaler_with_data(messages)

    def _load_dict_from_json(self, json_file):
        with open(json_file, 'r') as in_file:
            metrics_to_scaler_params_dict = json.load(in_file)
        return metrics_to_scaler_params_dict

    def _load_messages_from_file(self, name, delimiter='%'):
        messages = []
        with open(name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row_data in reader:
                json_data = str(row_data[0]).replace("\'", "\"")
                messages.append(State(json.loads(json_data)))
        return messages

    def _init_scaler_with_dict(self, metrics_mean_and_sum_sq):
        for metric, params in metrics_mean_and_sum_sq.items():
            scaler = self.scalers[metric]
            scaler.mean = params[0]
            scaler.sum_squares = params[1]
            scaler.nonzero = params[2]

    def serialise_scalers(self):
        metric_to_scaler_params = dict()
        save_file = os.path.join(
            self.experiment_config['model_generator_args']['json_dir'],
            self.experiment_config['model_generator_args']['json_file'])
        for metric, scaler in self.scalers.items():
            metric_to_scaler_params[metric] = scaler.get_params()
        serialise_string = json.dumps(metric_to_scaler_params,
                                      ensure_ascii=False, indent=4)
        with open(save_file, 'w+') as out_file:
            out_file.write('{}\n'.format(serialise_string))

    # messages is a massive list of system states
    def _init_scaler_with_data(self, messages):
        # get the keys of the data

        metrics_matrix = dict()
        for metric in self.metric_to_index.keys():
            metrics_matrix[metric] = []
            for state_message in messages:
                message = state_message.as_dict()
                # obtain the metric_to_index
                component_to_index = message['name_to_index']
                for bolt in self.bolts:
                    # sort them to be consistent
                    metrics_matrix[metric].append(
                        sorted(message['metrics'][metric] \
                                   [component_to_index[bolt]]))
        # metrics_matrix now shoul contain for each metric a list of all the 
        # instances obtained. Now just need to pad these with zeroes + 
        # feed to the appropriate standard scaler. 
        for metric, collections in metrics_matrix.items():
            padded_metric_data = np.zeros((len(collections),
                                           self.max_instances))
            for i in range(len(collections)):
                for j in range(len(collections[i])):
                    padded_metric_data[i][j] = collections[i][j]
            # feed these to the scaler
            self.scalers[metric].fit(padded_metric_data)

    def system_to_agent_state(self, message, use_max=False, add_to_scaler=False):
        # get the metrics
        metrics = message.as_dict()['metrics']
        agent_state = dict()
        # Add + 1 if using par
        agent_state['metrics'] = np.zeros((len(metrics.keys()), \
                                           self.max_instances))
        for metric, values in metrics.items():
            # sort the values 
            values_sorted = sorted(values)
            values_non_zero_sorted = []
            # filter out the zero elements
            for value in values_sorted:
                if value != 0.0:
                    values_non_zero_sorted.append(value)
            # copy to an appropriately sized numpy array
            values_array = np.zeros((self.max_instances,))
            for i in range(len(values_sorted)):
                values_array[i] = values_sorted[i]

            # normalise these according to the mean and covariance
            # reshape to allow single sample
            values_array = values_array.reshape(1, -1)
            if add_to_scaler:
                self.scalers[metric].partial_fit(values_array)

            state = self.scalers[metric].transform(values_array)
            self.logger.debug('Agent State: {}'.format(state))
            if not np.all(np.isfinite(state)):
                self.logger.warning('NaN or infinite values found in state for metric {}'
                                    ' in values_array {}'.format(metric, values_array))
            agent_state['metrics'][self.metric_to_index[metric], :] = state

        return State(agent_state)
