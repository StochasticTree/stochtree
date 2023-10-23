"""
Python frontend to the XBART C++ codebase
"""
import numpy as np
from stochtree_cpp import XBART_CPP
from .utils import _param_dict_to_str

class XBART:
    def __init__(self, num_trees = 100, min_data_in_leaf = 10, max_depth = -1, 
                 alpha = 0.9, beta = 2., a_sigma = 16, b_sigma = 4, a_tau = 3, 
                 b_tau = 0.5, data_driven_prior = True) -> None:
        # Set config parameters
        param_dict = {
            "num_trees": num_trees, 
            "min_data_in_leaf": min_data_in_leaf, 
            "max_depth": max_depth, 
            "alpha": alpha, "beta": beta, 
            "a_sigma": a_sigma, "b_sigma": b_sigma, 
            "a_tau": a_tau, "b_tau": b_tau, 
            "data_driven_prior": data_driven_prior
        }
        param_str = _param_dict_to_str(param_dict)

        # Initialize a BART_CPP object
        self.xbart_cpp = XBART_CPP(param_str)

    def sample(self, X: np.array, y: np.array, num_samples: int = 200, num_burnin: int = 100):
        """
        Run the XBART sampler for a number of draws
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
        self.xbart_cpp.reset_params(param_str)

        # Run the sampler
        self.xbart_cpp.sample(model_matrix, p, n)

    def predict(self, X: np.array):
        """
        Predict from the sampled XBART draws
        """
        # Concatenate the data into a contiguous numpy array
        n, p = X.shape
        model_matrix = np.concatenate((np.zeros((n, 1)), X), axis = 1)
        
        # Predict from the BART draws
        return self.xbart_cpp.predict(model_matrix, p, n)
