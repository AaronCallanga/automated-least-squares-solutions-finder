"""
Formatting Utilities Module

Provides functions to format numbers as decimals or fractions.
"""
import numpy as np
from fractions import Fraction
from typing import Union
import pandas as pd


def to_fraction_str(value: float, max_denominator: int = 1000) -> str:
    """
    Convert a float to a fraction string.
    
    Args:
        value: The float value to convert
        max_denominator: Maximum denominator for the fraction
    
    Returns:
        String representation of the fraction
    """
    if np.isnan(value) or np.isinf(value):
        return str(value)
    
    frac = Fraction(value).limit_denominator(max_denominator)
    
    if frac.denominator == 1:
        return str(frac.numerator)
    else:
        return f"{frac.numerator}/{frac.denominator}"


def format_array(arr: np.ndarray, use_fractions: bool = False) -> pd.DataFrame:
    """
    Format a numpy array for display.
    
    Args:
        arr: The numpy array to format
        use_fractions: If True, display as fractions; if False, as decimals
    
    Returns:
        Pandas DataFrame with formatted values
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    
    if use_fractions:
        formatted = np.vectorize(to_fraction_str)(arr)
        return pd.DataFrame(formatted)
    else:
        return pd.DataFrame(arr)


def format_value(value: float, use_fractions: bool = False) -> str:
    """
    Format a single value as decimal or fraction.
    
    Args:
        value: The value to format
        use_fractions: If True, display as fraction; if False, as decimal
    
    Returns:
        Formatted string
    """
    if use_fractions:
        return to_fraction_str(value)
    else:
        return f"{value:.6f}"
