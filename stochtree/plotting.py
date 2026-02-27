from typing import Union
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from .bart import BARTModel
from .bcf import BCFModel


def _validate_inputs(model: Union[BARTModel, BCFModel], term: str) -> None:
    """
    Validate the parameter name against the model's expected parameter names.
    """
    bart_valid_terms = [
        "sigma2", "global_error_scale", "sigma2_global",
        "sigma2_leaf", "leaf_scale", 
    ]
    bcf_valid_terms = [
        "sigma2", "global_error_scale", "sigma2_global",
        "sigma2_leaf_mu", "leaf_scale_mu", "mu_leaf_scale",
        "sigma2_leaf_tau", "leaf_scale_tau", "tau_leaf_scale",
        "adaptive_coding", 
    ]
    if not isinstance(model, BARTModel) and not isinstance(model, BCFModel):
        raise ValueError("Unsupported model type.")
    if not isinstance(term, str):
        raise ValueError("Term must be a string.")
    if isinstance(model, BARTModel):
        if term not in bart_valid_terms:
            raise ValueError(f"Invalid term '{term}' for model type {type(model).__name__}")
    elif isinstance(model, BCFModel):
        if term not in bcf_valid_terms:
            raise ValueError(f"Invalid term '{term}' for model type {type(model).__name__}")


def plot_parameter_trace(
    model: Union[BARTModel, BCFModel], 
    term: str, 
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot the parameter trace for a given model. For `BARTModel` objects, the following conventions are used for parameter names:
      - Global error variance: `"sigma2"`, `"global_error_scale"`, `"sigma2_global"`
      - Leaf scale: `"sigma2_leaf"`, `"leaf_scale"`
    For `BCFModel` objects, the following conventions are used for parameter names:
      - Global error variance: `"sigma2"`, `"global_error_scale"`, `"sigma2_global"`
      - Prognostic forest leaf scale: `"sigma2_leaf_mu"`, `"leaf_scale_mu"`, `"mu_leaf_scale"`
      - Treatment effect forest leaf scale: `"sigma2_leaf_tau"`, `"leaf_scale_tau"`, `"tau_leaf_scale"`
      - Adaptive coding parameters: `"adaptive_coding"` (returns both the control and treated parameters jointly, with control in the first row and treated in the second row)
    
    For traceplots / histograms of functional terms like `"y_hat_train"` or `"tau_hat_train"`, use the `model.extract_parameter_trace()` method to 
    query a (2d / 3d) parameter array and then plot directly using `pyplot.plot`, `pyplot.scatter`, or `pyplot.hist`.

    Parameters
    ----------
    model : Union[BARTModel, BCFModel]
        The model from which to plot a parameter trace.
    term : str
        The parameter or term to be plotted. If the term is not found in the model, an error will be raised.
    ax : plt.Axes, optional
        The matplotlib axes on which to plot the parameter trace. If not specified, a new figure and axes will be created.

    Returns
    -------
    plt.Axes
        The matplotlib axes object containing the parameter trace plot.
    """
    # Input validation
    _validate_inputs(model, term)

    # Extract parameter samples
    parameter_array = model.extract_parameter(term)

    # Squeeze and check dimensions
    parameter_array = np.squeeze(parameter_array)
    param_dim = parameter_array.ndim
    
    # Check cases
    if param_dim > 2:
        raise ValueError("Invalid parameter array shape.")

    # Initialize matplotlib fig and ax
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax = ax

    # Plot parameters
    if isinstance(model, BARTModel):
        ax.plot(parameter_array)
    elif isinstance(model, BCFModel):
        if term in ['adaptive_coding']:
            ax.plot(parameter_array[0,:], label='Control')
            ax.plot(parameter_array[1,:], label='Treated')
            ax.legend()
        else:
            ax.plot(parameter_array)

    ax.set_title(f"Parameter Trace: {term}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter Values")

    return ax
