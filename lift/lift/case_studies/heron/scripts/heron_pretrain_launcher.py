import sys
import argparse
import json
import logging

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                              datefmt='%y-%m-%d %H:%M:%S')
logger = logging.getLogger('')
logger.setLevel(logging.INFO)

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

logger.addHandler(sh)

import numpy as np
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from lift.case_studies.heron.heron_pretrain_helper import HeronPretrainController


def compute_stats(results):
    means = []
    stddevs = []
    for i in range(len(results[0])):
        mean = 0
        stddev = 0
        j = 0
        while j < len(results) and i < len(results[j]):
            oldmean = mean
            mean = mean + (results[j][i] - mean) / (j + 1)
            # actually computes sum of squared deviations
            stddev = stddev + (results[j][i] - mean) * \
                     (results[j][i] - oldmean)

            j += 1
        means.append(mean)
        div = j - 1 if j - 1 > 0 else 1
        stddevs.append(np.sqrt(stddev / div))
    return means, stddevs


def main(argv):
    parser = argparse.ArgumentParser('Heron Pretraining Script')
    parser.add_argument('experiment_config', help='JSON experiment config')
    parser.add_argument('agent_config', help='JSON agent config')
    parser.add_argument('--network-config', help='JSON network config')
    parser.add_argument('--model-path', help='Path to save the model to',
                        default='')
    parser.add_argument('--test-only', action='store_true',
                        help='Do not pretrain the model, only test it.')
    parser.add_argument('--repeats', default='1', help='Number of repeats to '
                                                       'do when pretraining')
    args = parser.parse_args()
    with open(args.agent_config, 'r') as agent_file:
        agent_config = json.load(agent_file)
    with open(args.experiment_config, 'r') as experiment_fd:
        experiment_config = json.load(experiment_fd)
    network_config = None
    if args.network_config:
        with open(args.network_config, 'r') as network_fd:
            network_config = json.load(network_fd)

    pretrain_results = []
    pretrain_train_results = []
    for i in range(int(args.repeats)):
        pretrain_helper = HeronPretrainController(agent_config, network_config,
                                                  experiment_config, model_path=args.model_path,
                                                  no_plot=(int(args.repeats) > 1))
        if not args.test_only:
            results, train_results = pretrain_helper.run(
                early_stopping=False)
            pretrain_results.append(results)
            pretrain_train_results.append(train_results)
        else:
            pretrain_helper.load_and_evaluate(args.model_path)

    # compute the error in the measurements
    pretrain_results.sort(key=len, reverse=True)
    pretrain_train_results.sort(key=len, reverse=True)

    # pretrain_means, pretrain_stddevs = compute_stats(pretrain_results)
    # train_means, train_stddevs = compute_stats(pretrain_train_results)
    pretrain_means = np.mean(pretrain_results, axis=0)
    pretrain_stddevs = np.std(pretrain_results, axis=0) / \
                       np.sqrt(len(pretrain_results[0]))
    train_means = np.mean(pretrain_train_results, axis=0)
    train_stddevs = np.std(pretrain_train_results, axis=0) / \
                    np.sqrt(len(pretrain_train_results[0]))
    # plot these bad boys 
    plt.figure()
    plt.errorbar([i for i in range(len(pretrain_means))], pretrain_means,
                 yerr=pretrain_stddevs, marker='+')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')

    plt.figure()
    plt.errorbar([i for i in range(len(train_means))], train_means,
                 yerr=train_stddevs, marker='+')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
