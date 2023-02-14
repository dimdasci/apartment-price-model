"""Provides feature and target transformations"""

import pandas as pd
import numpy as np

def transform_target(target: pd.Series) -> pd.Series:
    """Applies log10 to targer"""

    return np.log10(target)

def restore_target(target: pd.Series) -> pd.Series:
    """Powers 10 to target"""
    
    return 10**target