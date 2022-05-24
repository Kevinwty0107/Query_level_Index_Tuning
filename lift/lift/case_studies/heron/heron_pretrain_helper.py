import os
import time
from collections import deque
import numpy as np
import tensorflow as tf
from tensorforce.agents import Agent

from lift.pretraining.pretrain_controller import PretrainController
from lift.case_studies.heron.reward_plotter import RewardPlotter
from lift.case_studies.heron import heron_schemas
from lift.case_studies.heron.heron_data_loader import PretrainDataSource
from lift.case_studies.heron.plot_heatmap import HeatmapPlotter
from lift.case_studies.heron.heron_evaluator import HeronEvaluator


class HeronPretrainController(PretrainController):

    def __init__(
        self,
        agent_config,
        network_config,
        pretrain_config,
        result_dir=None,
        model_path=None,
        load_model=False,
        training_dir=None,
        test_dir=None,
        no_plot=False
    ):

        super(HeronPretrainController, self).__init__(model_path, load_model)
        self.pretrain_loader = PretrainDataSource(pretrain_config)
        self.trace_file = os.path.join(pretrain_config['demo_dir'],
                                       pretrain_config['demo_file'])
        self.epochs = pretrain_config['epochs']
        self.pretrain_batch_size = pretrain_config['batch_size']
        # get the schema to feed to the agent
        self.schema = heron_schemas[pretrain_config['schemas']](
            pretrain_config)
        self.logger.debug(self.schema)
        self.states_spec = self.schema.get_states_spec()
        self.action_spec = self.schema.get_actions_spec()
        if network_config:
            network_spec = network_config
        else:
            layer_size = pretrain_config['layer_size']
            network_spec = [
                dict(type='flatten'),
                dict(type='dense', size=layer_size),
                dict(type='dense', size=layer_size)
            ]
        # create the agent
        tf.reset_default_graph()
        # create the agent 
        self.agent = Agent.from_spec(
            spec=agent_config,
            kwargs=dict(
                states=self.states_spec,
                actions=self.action_spec,
                network=network_spec
            )
        )
        self.early_stopping_len = 15
        if not no_plot:
            self.plotter = RewardPlotter('/local/scratch/mks40/trace_benchmark/heron_results/plots/')
        self.no_plot = no_plot
        self.evaluator = HeronEvaluator(self.agent)
        self.train_evaluator = HeronEvaluator(self.agent)
        self.agent_time = 0
        self.evaluation_time = 0
        self.loading_data_time = 0
        self.total_time = 0

    def run(self, early_stopping=True):
        total_time_start = time.perf_counter()
        data_load_start = time.perf_counter()
        batch = self.pretrain_loader.load_data(self.trace_file, concat=False)

        # evaluation states
        states, actions, _, _ = self.pretrain_loader.get_evaluation_data()
        the_states = []

        for state in states:
            the_states.append(state.as_dict())
        # training set states
        the_train_states, train_actions, _, _ = self.pretrain_loader.load_data(
            self.trace_file, return_tuple=True)
        train_states = []
        for train_state in the_train_states:
            train_states.append(train_state.as_dict())

        # import the demonstrations
        self.agent.import_demonstrations(demonstrations=batch)
        data_load_end = time.perf_counter()
        # train the agent for a bunch of epochs
        best_accuracy = 0.0
        counter = 0
        best_agent_actions = []
        if early_stopping:
            last_5_accuracies = deque(maxlen=self.early_stopping_len)
            stop = False
            counter = 1
        for i in range(self.epochs):
            if early_stopping and stop:
                break
            self.logger.info("======EPOCH={}=========".format(i))
            iterations = int(np.ceil(len(batch) / self.pretrain_batch_size))
            agent_start = time.perf_counter()
            for j in range(iterations):
                self.agent.run(self.pretrain_batch_size)
            agent_end = time.perf_counter()
            # evaluate the results at the end
            evaluate_start = time.perf_counter()
            self.logger.info('---Training Set Evaluation----')
            self.train_evaluator.evaluate(states=train_states,
                                          actions=train_actions)
            self.logger.info('---Validation Set Evaluation----')
            accuracy = self.evaluator.evaluate(states=the_states,
                                               actions=actions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.agent.save_model(self.model_path + '/pretrain_model.ckpt', False)
                best_agent_actions = self.evaluator.agent_actions
            if early_stopping:
                oldest_acc = 0.0
                if len(last_5_accuracies) == self.early_stopping_len:
                    oldest_acc = last_5_accuracies.popleft()

                last_5_accuracies.append(accuracy)
                if len(last_5_accuracies) == self.early_stopping_len \
                        and oldest_acc >= max(last_5_accuracies):
                    stop = True
                else:
                    counter += 1
            evaluate_end = time.perf_counter()
            self.evaluation_time += (evaluate_end - evaluate_start)
            self.agent_time += (agent_end - agent_start)

        self.loading_data_time += (data_load_end - data_load_start)
        # plot the training history
        # get the most recent agent actions
        true_actions = []
        agent_actions = []
        for action in actions:
            true_actions.append(action['par'])
        for action in best_agent_actions:
            agent_actions.append(action['par'])
        if not self.no_plot:
            HeatmapPlotter.plot(true_actions, agent_actions, 7)
            self.plotter.plot_list(
                np.arange(len(self.evaluator.training_history)),
                self.evaluator.training_history, xlabel='Epochs',
                ylabel='Validation Accuracy'
            )
            self.plotter.plot_list(
                np.arange(len(self.train_evaluator.training_history)),
                self.train_evaluator.training_history, xlabel='Epochs',
                ylabel='Training Accuracy')
        total_time_end = time.perf_counter()
        self.total_time += (total_time_end - total_time_start)
        self.logger.info('Agent Training Time: {}'.format(self.agent_time))
        self.logger.info('Evaluation and Early Stopping Time: {}'.format(
            self.evaluation_time))
        self.logger.info('Loading Data Time: {}'.format(self.loading_data_time))
        self.logger.info('Total Time Taken: {}'.format(self.total_time))
        # if not self.no_plot:
        #     plt.show()
        return self.evaluator.training_history, self.train_evaluator.training_history

    def load_and_evaluate(self, model_path):
        states, actions, _, _ = self.pretrain_loader.get_evaluation_data()
        self.agent.restore_model(model_path)
        the_states = []
        for state in states:
            the_states.append(state.as_dict())
        accuracy = self.evaluator.evaluate(states=the_states, actions=actions)
        return accuracy
