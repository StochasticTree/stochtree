"""
Python frontend to the BART C++ codebase
"""
import numpy as np
from stochtree_cpp import BART_CPP
from .utils import _param_dict_to_str

class BART:
    def __init__(self, num_trees = 100, min_data_in_leaf = 10, max_depth = -1, 
                 alpha = 0.9, beta = 2., nu = 5, lam = 5, mu_mean = 0, mu_sigma = 1, 
                 data_driven_prior = True) -> None:
        # Set config parameters
        param_dict = {
            "num_trees": num_trees, 
            "min_data_in_leaf": min_data_in_leaf, 
            "max_depth": max_depth, 
            "alpha": alpha, "beta": beta, 
            "nu": nu, "lam": lam, 
            "mu_mean": mu_mean, 
            "mu_sigma": mu_sigma, 
            "data_driven_prior": data_driven_prior
        }
        param_str = _param_dict_to_str(param_dict)

        # Initialize a BART_CPP object
        self.bart_cpp = BART_CPP(param_str)

    def sample(self, X: np.array, y: np.array, num_samples: int = 200, num_burnin: int = 100):
        """
        Run the BART sampler for a number of draws
        """
        # Concatenate the data into a contiguous numpy array
        n, p = X.shape
        model_matrix = np.concatenate((np.expand_dims(y, axis=1), X), axis = 1)

        # Set config parameters
        param_dict = {
            "num_samples": num_samples, 
            "num_burnin": num_burnin, 
            "label_column": 0
        }
        param_str = _param_dict_to_str(param_dict)
        self.bart_cpp.reset_params(param_str)

        # Run the sampler
        self.bart_cpp.sample(model_matrix, p, n)

    def predict(self, X: np.array):
        """
        Predict from the sampled BART draws
        """
        # Concatenate the data into a contiguous numpy array
        n, p = X.shape
        model_matrix = np.concatenate((np.zeros((n, 1)), X), axis = 1)
        
        # Predict from the BART draws
        return self.bart_cpp.predict(model_matrix, p, n)
