"""
Input Components Module

Handles all user input forms and parsing for the Least Squares Calculator.
Provides two input modes:
    1. Data Points (Easy) - Enter x,y coordinates for line fitting
    2. Matrix Input (Advanced) - Enter matrix A and vector b using editable tables
"""
import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def render_data_points_input() -> Tuple[str, str]:
    """
    Render the data points input form (Easy mode).
    
    Returns:
        Tuple of (x_input, y_input) as raw strings
    """
    st.header("Enter Your Data Points")
    st.info("**Tip:** Enter x and y coordinates to find the best-fit line: y = mx + c")
    
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


def render_matrix_input() -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Render the matrix input form using editable tables (Advanced mode).
    
    Returns:
        Tuple of (A, b, error_message)
    """
    st.header("Enter Matrix A and Vector b")
    st.latex(r"A\hat{x} = \vec{b}")
    st.info("**Tip:** Edit the cells directly. Use the controls to adjust matrix dimensions.")
    
    # Matrix dimension controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        num_rows = st.number_input("Number of rows (m)", min_value=2, max_value=10, value=5, step=1)
    with col2:
        num_cols = st.number_input("Number of columns (n)", min_value=1, max_value=5, value=2, step=1)
    
    st.markdown("---")
    
    # Create default data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.latex(r"A =")
        
        # Create default matrix A data
        default_A = []
        for i in range(int(num_rows)):
            row = [float(i + 1)] + [1.0] * (int(num_cols) - 1)
            default_A.append(row[:int(num_cols)])
        
        # Column names for matrix A
        col_names = [f"a{j+1}" for j in range(int(num_cols))]
        
        df_A = pd.DataFrame(default_A, columns=col_names)
        
        # Editable dataframe for matrix A
        edited_A = st.data_editor(
            df_A,
            num_rows="fixed",
            use_container_width=False,
            hide_index=True,
            key="matrix_A"
        )
    
    with col2:
        st.latex(r"\vec{b} =")
        
        # Create default vector b data
        default_b = [[2.1], [4.0], [5.8], [8.1], [9.9]]
        # Adjust to match number of rows
        while len(default_b) < int(num_rows):
            default_b.append([0.0])
        default_b = default_b[:int(num_rows)]
        
        df_b = pd.DataFrame(default_b, columns=["b"])
        
        # Editable dataframe for vector b
        edited_b = st.data_editor(
            df_b,
            num_rows="fixed",
            use_container_width=False,
            hide_index=True,
            key="vector_b"
        )
    
    # Convert to numpy arrays
    try:
        A = edited_A.values.astype(float)
        b = edited_b.values.flatten().astype(float)
        
        # Validation
        if A.shape[0] != len(b):
            return None, None, f"Matrix A has {A.shape[0]} rows but vector b has {len(b)} elements. They must match!"
        
        return A, b, ""
    
    except ValueError as e:
        return None, None, f"Invalid input! Please enter numbers only. Error: {e}"


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
    This function is kept for backwards compatibility but is not used in table mode.
    """
    try:
        A = np.array([
            [float(val) for val in row.split()] 
            for row in a_input.strip().split('\n') 
            if row.strip()
        ])
        
        b = np.array([
            float(val.strip()) 
            for val in b_input.strip().split('\n') 
            if val.strip()
        ])
        
        if A.shape[0] != len(b):
            return None, None, f"Matrix A has {A.shape[0]} rows but vector b has {len(b)} elements. They must match!"
        
        return A, b, ""
    
    except ValueError as e:
        return None, None, f"Invalid input! Please check your matrix format. Error: {e}"
