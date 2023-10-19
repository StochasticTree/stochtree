"""
Utility functions for the python library, many of which come from LightGBM's basic.py module
"""
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

_NUMERIC_TYPES = (int, float, bool)

def _param_dict_to_str(data: Optional[Dict[str, Any]]) -> str:
    """Convert Python dictionary to string for C++ configuration
    
    From LightGBM's python package:
    https://github.com/microsoft/LightGBM/blob/8ed371cee49cf86740b25dd9a4b985a75c9f2dba/python-package/lightgbm/basic.py#L431
    """
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or _is_numpy_1d_array(val):
            pairs.append(f"{key}={','.join(map(_to_string, val))}")
        elif isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val):
            if key == "lam":
                pairs.append(f"lambda={val}")
            else:
                pairs.append(f"{key}={val}")
        elif val is not None:
            raise TypeError(f'Unknown type of parameter:{key}, got:{type(val).__name__}')
    return ' '.join(pairs)

def _is_numeric(obj: Any) -> bool:
    """Check whether object is a number or not, include numpy number, etc.
    
    From LightGBM's python package:
    https://github.com/microsoft/LightGBM/blob/8ed371cee49cf86740b25dd9a4b985a75c9f2dba/python-package/lightgbm/basic.py#L253C1-L261C21
    """
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: invalid literal
        return False

def _is_numpy_1d_array(data: Any) -> bool:
    """Check whether data is a numpy 1-D array.
    
    From LightGBM's python package:
    https://github.com/microsoft/LightGBM/blob/8ed371cee49cf86740b25dd9a4b985a75c9f2dba/python-package/lightgbm/basic.py#L264C1-L267C1
    """
    return isinstance(data, np.ndarray) and len(data.shape) == 1

def _to_string(x: Union[int, float, str, List]) -> str:
    """
    From LightGBM's python package:
    https://github.com/microsoft/LightGBM/blob/8ed371cee49cf86740b25dd9a4b985a75c9f2dba/python-package/lightgbm/basic.py#L423C1-L428C22
    """
    if isinstance(x, list):
        val_list = ",".join(str(val) for val in x)
        return f"[{val_list}]"
    else:
        return str(x)
