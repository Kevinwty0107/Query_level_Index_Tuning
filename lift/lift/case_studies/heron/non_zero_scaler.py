import numpy as np

class NonZeroScaler(object):

    def __init__(self):
        self.mean = None
        self.sum_squares = None
        self.nonzero = None

    def fit(self, X_list):
        X = np.array(X_list)
        n_samples, n_features = X.shape
        # compute the mean of all these vectors init making sure to ignore 0s
        X_sum = np.sum(X, axis = 0)
        self.nonzero = np.count_nonzero(X, axis = 0)
        # Divide these two element-wise
        self.mean = X_sum / self.nonzero
        # convert nan to num
        np.nan_to_num(self.mean, copy=False)
        # compute the variance
        self.sum_squares = np.sum((X - self.mean) ** 2, axis=0)
      
    def get_params(self):
        return [self.mean.tolist(), self.sum_squares.tolist(), 
                self.nonzero.tolist()]

    def transform(self, X_list):
        X = np.array(X_list)
        factor = np.ones_like(self.mean) / (self.nonzero - 1)
        stddev = np.sqrt(self.sum_squares * factor)
        # need to scale so that only the parts where X is not 
        # zero are scaled.
        ret = np.squeeze((X - self.mean)) / stddev
        # set the areas that are zero in the input to zero in the output
        zero_rows, zero_cols = np.where(X == 0.0)
        ret[zero_cols] = 0.0
        return np.nan_to_num(ret, copy=False)

    def partial_fit(self, X_list):
        # assume X contains a single sample
        X = np.array(X_list)
        nonzero = np.count_nonzero(X, axis = 0)
        oldmean = np.copy(self.mean)
        self.nonzero += nonzero
        self.mean = self.mean + \
                np.nan_to_num(np.squeeze((X - self.mean) / self.nonzero))
        self.sum_squares = self.sum_squares + \
                np.squeeze((X - oldmean) * (X - self.mean))
        

