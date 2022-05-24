import logging
import time
import numpy as np
from lift.case_studies.mongodb.fixed_imdb_workload import FixedIMDBWorkload

from lift.rl_model.task_graph import TaskGraph
from lift.util.parsing_util import sort_input


class PretrainController(object):
    """
    Generic pretrain controller.

    Processes workload data to augment states with actions and rewards using
    rule-based or model-based labeling.
    """

    def __init__(self, agent_config, experiment_config, model_path, load_model, result_dir, blackbox_mode):
        self.logger = logging.getLogger(__name__)

        self.agent_config = agent_config
        self.experiment_config = experiment_config
        self.result_dir = result_dir
        self.blackbox_mode = blackbox_mode
        self.training_baseline_label = experiment_config.get('pretraining_baseline_name', 'field_matching_')

        self.rewards_available = experiment_config.get('rewards_available', False)
        self.pretrain_steps = experiment_config['pretrain_steps']
        self.pretrain_batch_size = experiment_config['pretrain_batch_size']

        self.rewards_available = experiment_config.get('rewards_available', False)
        self.next_states = experiment_config.get("next_states", True)
        # Do not eval on default  -> takes too much time.
        self.evaluate_accuracy = experiment_config.get("evaluate_accuracy", False)
        self.custom_margin = experiment_config["custom_margin"]
        self.state_mode = experiment_config.get("state_mode", "default")

        self.model_path = model_path
        self.load_model = load_model

        self.workload_gen = None
        self.actions_spec = None
        self.states_spec = None

        self.queries = None

        self.converter = None
        self.demo_rules = None

        self.evaluator = None

        # Empty default task graph.
        self.task_graph = TaskGraph()
        self.agent_type = None

    def run(self, *args, **kwargs):
        """
        Executes pre-training.
        """
        batch, margins, pretraining_io_time = self.process_trace()
        self.pretrain(batch, margins, pretraining_io_time)

    def pretrain(self, batch, margins, pretraining_io_time):
        self.observe(batch)
        start = time.monotonic()

        if not self.custom_margin:
            margins = None
        losses = []
        for i in range(self.pretrain_steps):
            if i % 100 == 0:
                self.logger.info("Beginning update  {}.".format(i))
            # Sub-sample -> Do not sub-sample in blackbox mode.
            if self.pretrain_batch_size > 0 and self.blackbox_mode is False:
                sample_batch, sample_margins = self.random_batch(batch=batch, margins=margins,
                                                                 batch_size=self.pretrain_batch_size)
            else:
                sample_batch = batch
                sample_margins = margins
            # Update from external batch, apply demo loss, and pass custom margins if configured.
            loss, _ = self.task_graph.get_task("").unwrap().\
                update(batch=sample_batch,
                       update_from_demos=False,
                       apply_demo_loss_to_batch=True,
                       expert_margins=sample_margins)
            losses.append(loss)
        # Time actually spent updating.
        total_training_time = time.monotonic() - start

        # Store model.
        self.task_graph.store_model(name="", path=self.model_path)
        np.savetxt(fname=self.result_dir + '/timing/pretraining_times.txt',
                   X=np.asarray([total_training_time, pretraining_io_time]),delimiter=',')
        np.savetxt(fname=self.result_dir + '/pretrain_losses.txt',
                   X=np.asarray(losses),delimiter=',')

    def process_trace(self):
        """
        Prepares off-policy training data by fetching traces from a workload generator, and enriching
        them using one or more demonstration rules.
        """
        start = time.monotonic()
        # 1. Generate queries
        num_episodes = self.experiment_config["num_demo_episodes"]
        # 2. Generate demos using queries and demo rules.
        # Chunk into episodes where each episode has its own context.
        actions = {}
        for name in self.actions_spec.keys():
            actions[name] = []
        if self.state_mode == 'index_net':
            states = {}
            next_states = {}
            for name in self.states_spec.keys():
                states[name] = []
                next_states[name] = []
        else:
            states = []
            next_states = []
        batch = dict(
            states=states,
            actions=actions,
            terminals=[],
            next_states=next_states,
            rewards=[]
        )
        # Custom margins.
        # Generate or get all queries.
        num_demo_queries = self.experiment_config["num_demo_queries"]

        # In black-box mode, ensure only a single episode.
        if self.blackbox_mode:
            self.logger.info("Blackbox mode enabled, setting num_episodes to 1, was {}".format(num_episodes))
            num_episodes = 1
        self.queries = []
        for _ in range(num_episodes):
            self.queries.extend(self.workload_gen.define_demo_queries(num_demo_queries))

        demo_margins = []
        for demo_rule in self.demo_rules:
            for ep in range(num_episodes):
                # Get queries for episode slice, create episode data.
                ep_queries = self.queries[ep * num_demo_queries:(ep + 1) * num_demo_queries]
                ep_queries = sort_input(ep_queries)

                # Begin episode with empty context.
                episode_context = []
                for i, query in enumerate(ep_queries):
                    # Convert query.
                    state = self.converter.system_to_agent_state(query,
                                                                 system_context=dict(index_columns=episode_context))

                    # TODO, how do handle human demos in a workflow?
                    if isinstance(self.workload_gen, FixedIMDBWorkload):
                        # Demo is already annotated on query.
                        system_action = query.demonstration()
                    else:
                        # Note, using query here because more convenient representation than state.
                        system_action = demo_rule.generate_demonstration(states=query, context=episode_context)
                    if self.custom_margin:
                        demo_margins.append(demo_rule.margin())
                    agent_action = self.converter.system_to_agent_action(system_action=system_action, query=query)

                    if self.state_mode == 'index_net':
                        for name, value in state.get_value().items():
                            batch["states"][name].append(value)
                    else:
                        batch["states"].append(state.get_value())
                    for name, value in agent_action.items():
                        batch["actions"][name].append(value)
                    batch["rewards"].append(demo_rule.reward())
                    batch["terminals"].append(False)

                    # Update context.
                    episode_context.append(system_action["index"])

                    if self.next_states:
                        # Last next query is just the last query again.
                        next_query = ep_queries[i + 1] if i < len(ep_queries) - 1 else ep_queries[-1]
                        next_state = self.converter.system_to_agent_state(next_query,
                                                                          system_context=dict(index_columns=episode_context))
                        if self.state_mode == 'index_net':
                            for name, value in next_state.get_value().items():
                                batch["next_states"][name].append(value)
                        else:
                            batch["next_states"].append(next_state.get_value())
                # End current episode.
                batch["terminals"][-1] = True

        # Squeeze actions to remove extra dim.
        for name in self.actions_spec.keys():
            batch["actions"][name] = np.squeeze(np.asarray(batch["actions"][name]))
        if self.state_mode == "index_net":
            # Squeeze states to remove extra dim.
            for name in self.states_spec.keys():
                batch["states"][name] = np.squeeze(np.asarray(batch["states"][name]))
                batch["next_states"][name] = np.squeeze(np.asarray(batch["next_states"][name]))

        batch["importance_weights"] = np.ones_like(batch["rewards"])
        pretraining_io_time = time.monotonic() - start
        return batch, demo_margins, pretraining_io_time

    def observe(self, batch):
        """
        Calls observe function depending on agent type (i.e. extra demonstration memory), and
        considering if next_states are needed (value-based vs policy-based).

        Args:
            batch (dict): Demo batch.

        """
        if self.agent_type == "dqfd":
            self.logger.info("Observing demos for dqfd.")
            self.task_graph.get_task("").unwrap().observe_demos(
                preprocessed_states=batch["states"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_states=batch["next_states"],
                terminals=batch["terminals"]

            )
        elif self.agent_type in ["ppo", "actor_critic"]:
            self.logger.info("Observing demos for type {}.".format(self.agent_type))
            self.task_graph.observe_task(
                name="",
                preprocessed_states=batch["states"],
                actions=batch["actions"],
                internals=[],
                rewards=batch["rewards"],
                terminals=batch["terminals"]
            )
        elif self.agent_type in ["dqn", "sac"]:
            self.logger.info("Observing demos for type {}.".format(self.agent_type))
            self.task_graph.observe_task(
                name="",
                preprocessed_states=batch["states"],
                actions=batch["actions"],
                internals=[],
                rewards=batch["rewards"],
                next_states=batch["next_states"],
                terminals=batch["terminals"]
            )

    def random_batch(self, batch, margins, batch_size):
        data_size = len(batch["rewards"])
        indices = np.random.choice(np.arange(0, data_size - 1), size=batch_size)

        actions = {}
        for name in self.actions_spec.keys():
            actions[name] = []
        if self.state_mode == 'index_net':
            states = {}
            next_states = {}
            for name in self.states_spec.keys():
                states[name] = []
                next_states[name] = []
        else:
            states = []
            next_states = []
        sample_batch = dict(
            states=states,
            actions=actions,
            terminals=[],
            next_states=next_states,
            rewards=[]
        )
        sample_margins = []
        for index in indices:
            if self.state_mode == 'index_net':
                for name in self.states_spec.keys():
                    sample_batch["states"][name].append(batch["states"][name][index])
                    sample_batch["next_states"][name].append(batch["next_states"][name][index])
            else:
                sample_batch["states"].append(batch["states"][index])
                sample_batch["next_states"].append(batch["next_states"][index])

            for name in self.actions_spec.keys():
                sample_batch["actions"][name].append(batch["actions"][name][index])

            sample_batch["terminals"].append(batch["terminals"][index])
            sample_batch["rewards"].append(batch["rewards"][index])
            sample_margins.append(margins[index])

        for name in self.actions_spec.keys():
            sample_batch["actions"][name] = np.squeeze(np.asarray(sample_batch["actions"][name]))
        if self.state_mode == "index_net":
            # Squeeze states to remove extra dim.
            for name in self.states_spec.keys():
                sample_batch["states"][name] = np.squeeze(np.asarray(sample_batch["states"][name]))
                sample_batch["next_states"][name] = np.squeeze(np.asarray(sample_batch["next_states"][name]))

        sample_batch["importance_weights"] = np.ones_like(sample_batch["rewards"])
        return sample_batch, sample_margins
