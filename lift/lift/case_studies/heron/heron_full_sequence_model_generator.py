import os
import csv
import json
import numpy as np
from lift.case_studies.heron.non_zero_scaler import NonZeroScaler
from lift.case_studies.heron.heron_agent_model_generator import \
        HeronAgentModelGenerator
from lift.model.state import State


class HeronFullSequenceModelGenerator(HeronAgentModelGenerator):

    def __init__(self, constant_state, latency_normaliser,
            throughput_normaliser, reward_generator, experiment_config):
        super(HeronFullSequenceModelGenerator, self).__init__(constant_state,
                latency_normaliser, throughput_normaliser, reward_generator,
                experiment_config)
        self.max_instances = self.experiment_config['max_instances']
        self.bolts = self.experiment_config['bolts']
        load_from_json = self.experiment_config['model_generator_args']\
                ['load_from_json']
        
        # need a separate scaler for every metric 
        self.scaler = NonZeroScaler()
        if load_from_json:
            json_file = os.path.join(
                    self.experiment_config['model_generator_args']['json_dir'],
                    self.experiment_config['model_generator_args']['json_file']
            )
            params = self._load_params_from_json(json_file)
            self._init_scaler_with_params(params)
        else:
            message_file = os.path.join(self.experiment_config['demo_dir'],
                self.experiment_config['demo_file'])
            messages = self._load_messages_from_file(message_file)
            self._init_scaler_with_data(messages) 
    
    def _load_params_from_json(self, json_file):
        with open(json_file, 'r') as in_file:
             params = json.load(in_file)
        return params
    
    def _load_messages_from_file(self, name, delimiter='%'):
        messages = []
        with open(name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row_data in reader:
                messages.append(State(json.loads(row_data[0])))
        return messages
        
    def _init_scaler_with_params(self, params):
        self.scaler.mean = params[0]
        self.scaler.sum_squares = params[1]
        self.scaler.nonzero = params[2]
            
    def serialise_scalers(self):
        save_file = os.path.join(
                self.experiment_config['model_generator_args']['json_dir'],
                self.experiment_config['model_generator_args']['json_file'])
        params = self.scaler.get_params()
        serialise_string = json.dumps(params, 
                ensure_ascii = False, indent = 4)
        with open(save_file, 'w+') as out_file:
            out_file.write('{}\n'.format(serialise_string))


    # messages is a massive list of system states
    def _init_scaler_with_data(self, messages):
        # add the words to a large list
        words = []
        for message in messages:
            name_to_index = message.as_dict()['name_to_index']
            metrics = message.as_dict()['metrics']
            metric_to_values = dict()
            for metric, values in metrics.items():
                metric_to_values[metric] = []
                for bolt in self.bolts:
                    metric_to_values[metric].extend(values[name_to_index[bolt]])
            sequence = []
            # initialise the sequence
            for values in metric_to_values.values():
                for i in range(len(values)):
                    sequence.append(np.zeros((len(metric_to_values),)))
                break

            for metric, values in metric_to_values.items():
                print('Length of sequence: {}'.format(len(sequence)))
                print('Length of values: {}'.format(len(values)))
                for i in range(len(values)):
                    sequence[i][self.metric_to_index[metric]] = values[i]
            words.extend(sequence)
        
        # run the scaler on this list
        self.scaler.fit(words)
        
    def system_to_agent_state(self, message, use_max=False, add_to_scaler=False):
        # get the metrics
        metrics = message.as_dict()['metrics']
        agent_state = dict()
        # Add + 1 if using par
        agent_state['metrics'] = np.zeros((len(metrics.keys()),\
                self.max_instances))
        for metric, values in metrics.items():
            # put all the values in the rows
            for i in range(len(values)):
                agent_state['metrics'][self.metric_to_index[metric], i] = \
                        values[i]
        # take the transpose of the metrics
        agent_state['metrics'] = np.transpose(agent_state['metrics'])
        agent_state['metrics'] = agent_state['metrics'].tolist()
        # sort the list in python
        agent_state['metrics'].sort()
        scaled_metrics = []
        for metric_vector in agent_state['metrics']:
            metric_array = np.array(metric_vector).reshape(1, -1)
            if add_to_scaler:
                self.scaler.partial_fit(metric_array)
            scaled_metrics.append(self.scaler.transform(metric_array))
        # now convert back to a numpy array
        agent_state['metrics'] = np.transpose(np.array(scaled_metrics))
            
        return State(agent_state)
