import numpy as np
import json


# TODO DEPRECATED, old experiments
class MongoPretrainExporter(object):
    """
    Export helper for pretraining.
    """
    def export_search_results(self, path, final_dba_train_accuracies, final_default_train_accuracies,
                              final_default_test_accuracies, best_final_test_accuracy, best_run_index, best_config):

        final_default_test_accuracies.append(best_final_test_accuracy)
        final_default_test_accuracies.append(best_run_index)

        # Default Test
        final_test_results = np.asarray(final_default_test_accuracies)
        result_path = path + '/opt_default_test_accuracy_result.txt'
        np.savetxt(result_path, final_test_results, delimiter=',')

        # Default train
        final_default_train_results = np.asarray(final_default_train_accuracies)
        result_path = path + '/opt_default_train_accuracy_result.txt'
        np.savetxt(result_path, final_default_train_results, delimiter=',')

        # Dba Train
        final_dba_train_results = np.asarray(final_dba_train_accuracies)
        result_path = path + '/opt_dba_train_accuracy_result.txt'
        np.savetxt(result_path, final_dba_train_results, delimiter=',')

        config_path = path + '/best_config.json'

        json.dump(str(best_config), open(config_path, 'w'))

    def export_recommendation(self, path, dba_train_accuracy, default_train_accuracy,
                              default_test_accuracy, iteration=0, agent_config=None):
        # Evaluate on dba training queries
        dba_train_accuracies = np.asarray(dba_train_accuracy)
        recommendations_path = path + '/dba_train_accuracies' + str(iteration) + '.txt'
        np.savetxt(recommendations_path, dba_train_accuracies, delimiter=',')

        # Evaluate on default training queries
        default_train_accuracies = np.asarray(default_train_accuracy)
        recommendations_path = path + '/default_train_accuracies' + str(iteration) + '.txt'
        np.savetxt(recommendations_path, default_train_accuracies, delimiter=',')

        # Evaluate on default test queries
        default_test_accuracies = np.asarray(default_test_accuracy)
        recommendations_path = path + '/default_test_accuracies' + str(iteration) + '.txt'
        np.savetxt(recommendations_path, default_test_accuracies, delimiter=',')

        # Dump config
        config_path = path + '/agent_config_' + str(iteration) + '.json'

        json.dump(str(agent_config), open(config_path, 'w'))

    def export_plot_results(self, path, dba_train_a, dba_train_r, dba_train_p, default_train_a, default_train_r,
                            default_train_p, default_test_a, default_test_r, default_test_p):

        np.savetxt(path + '/dba_train_accuracies_means.csv', np.mean(dba_train_a, axis=0), delimiter=',')
        np.savetxt(path + '/dba_train_recalls_means.csv',  np.mean(dba_train_r, axis=0), delimiter=',')
        np.savetxt(path + '/dba_train_precisions_means.csv', np.mean(dba_train_p, axis=0), delimiter=',')
        np.savetxt(path + '/dba_train_accuracies_stds.csv', np.std(dba_train_a, axis=0), delimiter=',')
        np.savetxt(path + '/dba_train_recalls_stds.csv', np.std(dba_train_r, axis=0), delimiter=',')
        np.savetxt(path + '/dba_train_precisions_stds.csv',  np.std(dba_train_p, axis=0), delimiter=',')

        np.savetxt(path + '/default_train_accuracies_means.csv', np.mean(default_train_a, axis=0), delimiter=',')
        np.savetxt(path + '/default_train_recalls_means.csv', np.mean(default_train_r, axis=0), delimiter=',')
        np.savetxt(path + '/default_train_precisions_means.csv', np.mean(default_train_p, axis=0), delimiter=',')
        np.savetxt(path + '/default_train_accuracies_stds.csv', np.std(default_train_a, axis=0), delimiter=',')
        np.savetxt(path + '/default_train_recalls_stds.csv', np.std(default_train_r, axis=0), delimiter=',')
        np.savetxt(path + '/default_train_precisions_stds.csv', np.std(default_train_p, axis=0), delimiter=',')

        np.savetxt(path + '/default_test_accuracies_means.csv', np.mean(default_test_a, axis=0), delimiter=',')
        np.savetxt(path + '/default_test_recalls_means.csv', np.mean(default_test_r, axis=0), delimiter=',')
        np.savetxt(path + '/default_test_precisions_means.csv', np.mean(default_test_p, axis=0), delimiter=',')
        np.savetxt(path + '/default_test_accuracies_stds.csv', np.std(default_test_a, axis=0), delimiter=',')
        np.savetxt(path + '/default_test_recalls_stds.csv', np.std(default_test_r, axis=0), delimiter=',')
        np.savetxt(path + '/default_test_precisions_stds.csv', np.std(default_test_p, axis=0), delimiter=',')
