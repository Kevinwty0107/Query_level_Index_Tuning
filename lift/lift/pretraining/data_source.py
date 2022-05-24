import csv


class DataSource(object):
    """
    Generic data loader. Creates pre-train batches from off-policy data
    """

    def load_data(self, data_dir, **kwargs):
        """
        Restores serialised trajectories.

        Args:
            data_dir (str): Path to serialized dir.

        Returns:
             any: A batch of pretraining data.
        """
        raise NotImplementedError

    def export_data(self, data, data_dir, **kwargs):
        """
        Serialises trajectories.

        Args:
            data (any): Trajectory data.
            data_dir (str): Path to serialized dir.
        """
        pass

    def get_evaluation_data(self, **kwargs):
        """
        Returns evaluation data pairs of inputs and correct actions in case of
        expert training data.

        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets all loaded state.
        """
        pass

    @staticmethod
    def load_csv(path, delimiter='%'):
        parsed_rows = []
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row_data in reader:
                parsed_rows.append(row_data)
        return parsed_rows
