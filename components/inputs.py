"""
Input Components Module

Handles all user input forms and parsing for the Least Squares Calculator.
Provides two input modes:
    1. Data Points (Easy) - Enter x,y coordinates for line fitting
    2. Matrix Input (Advanced) - Enter matrix A and vector b directly
"""
import streamlit as st
import numpy as np
from typing import Tuple, Optional


def render_data_points_input() -> Tuple[str, str]:
    """
    Render the data points input form (Easy mode).
    
    Returns:
        Tuple of (x_input, y_input) as raw strings
    """
    st.header("📊 Enter Your Data Points")
    st.info("💡 **Tip:** Enter x and y coordinates to find the best-fit line: y = mx + c")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("X values (independent variable)")
        x_input = st.text_area(
            "Enter x values (one per line or comma-separated):",
            value="1\n2\n3\n4\n5",
            height=150,
            help="These are your input/independent values"
        )
    
    with col2:
        st.subheader("Y values (dependent variable)")
        y_input = st.text_area(
            "Enter y values (one per line or comma-separated):",
            value="2.1\n4.0\n5.8\n8.1\n9.9",
            height=150,
            help="These are your output/dependent values"
        )
    
    return x_input, y_input


def render_matrix_input() -> Tuple[str, str]:
    """
    Render the matrix input form (Advanced mode).
    
    Returns:
        Tuple of (a_input, b_input) as raw strings
    """
    st.header("🔢 Enter Matrix A and Vector b")
    st.info("💡 **Tip:** Solve Ax = b where A has more rows than columns (overdetermined)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Matrix A")
        a_input = st.text_area(
            "Enter matrix A (space-separated values, one row per line):",
            value="1 1\n2 1\n3 1\n4 1\n5 1",
            height=150,
            help="Each row is a separate line, values separated by spaces"
        )
    
    with col2:
        st.subheader("Vector b")
        b_input = st.text_area(
            "Enter vector b (one value per line):",
            value="2.1\n4.0\n5.8\n8.1\n9.9",
            height=150,
            help="One value per line"
        )
    
    return a_input, b_input


def parse_data_points(x_input: str, y_input: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Parse x and y input strings into numpy arrays.
    
    Args:
        x_input: Raw string of x values
        y_input: Raw string of y values
    
    Returns:
        Tuple of (x_values, y_values, error_message)
        If successful, error_message is empty.
        If failed, arrays are None and error_message contains the error.
    """
    try:
        # Parse x values (handle both newline and comma separation)
        x_values = np.array([
            float(x.strip()) 
            for x in x_input.replace(',', '\n').split('\n') 
            if x.strip()
        ])
        
        # Parse y values
        y_values = np.array([
            float(y.strip()) 
            for y in y_input.replace(',', '\n').split('\n') 
            if y.strip()
        ])
        
        # Validation
        if len(x_values) != len(y_values):
            return None, None, "X and Y must have the same number of values!"
        
        if len(x_values) < 2:
            return None, None, "Please enter at least 2 data points!"
        
        return x_values, y_values, ""
    
    except ValueError as e:
        return None, None, f"Invalid input! Please enter numbers only. Error: {e}"


def parse_matrix_input(a_input: str, b_input: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Parse matrix A and vector b input strings into numpy arrays.
    
    Args:
        a_input: Raw string of matrix A (space-separated, rows on new lines)
        b_input: Raw string of vector b (one value per line)
    
    Returns:
        Tuple of (A, b, error_message)
        If successful, error_message is empty.
        If failed, arrays are None and error_message contains the error.
    """
    try:
        # Parse matrix A
        A = np.array([
            [float(val) for val in row.split()] 
            for row in a_input.strip().split('\n') 
            if row.strip()
        ])
        
        # Parse vector b
        b = np.array([
            float(val.strip()) 
            for val in b_input.strip().split('\n') 
            if val.strip()
        ])
        
        # Validation
        if A.shape[0] != len(b):
            return None, None, f"Matrix A has {A.shape[0]} rows but vector b has {len(b)} elements. They must match!"
        
        return A, b, ""
    
    except ValueError as e:
        return None, None, f"Invalid input! Please check your matrix format. Error: {e}"
