from random import randint
from lift.case_studies.heron.load_gen import LoadGenerator
from importlib import import_module


class CombinedLoadGenerator(LoadGenerator):
    heron = import_module('lift.case_studies.heron')
    heron_load_generators = getattr(heron, 'heron_load_generators')

    def __init__(self, load_config, parallelism):
        super(CombinedLoadGenerator, self).__init__(load_config,
                                                    parallelism)
        # initialise all the load generators
        load_generators = load_config['load_generator_args']['load_generators']

        self.deterministic = load_config.get('deterministic', False)
        self.logger.info("Initializing load gen with deterministic mode = {}".format(
            self.deterministic
        ))
        self.load_generators = []
        for load_generator in load_generators:
            self.load_generators.append(
                self.heron_load_generators[load_generator](load_config,
                                                           parallelism))

    def loads(self):
        # pick a load_generator at random
        if self.deterministic:
            for load_generator in self.load_generators:
                for load in load_generator.loads():
                    yield load
        else:
            index = randint(0, len(self.load_generators) - 1)
            for load in self.load_generators[index].loads():
                yield load
