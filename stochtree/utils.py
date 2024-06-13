class NotSampledError(ValueError, AttributeError):
    """Exception class to raise if attempting to predict from a model before it has been sampled.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Renamed from scikit-learn's "NotFittedError" 
    https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/exceptions.py#L45C7-L45C21
    """