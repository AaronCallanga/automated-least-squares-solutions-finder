"""
Input Components Module

Handles all user input forms and parsing for the Least Squares Calculator.
Provides two input modes:
    1. Data Points (Easy) - Enter x,y coordinates for line fitting
    2. Matrix Input (Advanced) - Enter matrix A and vector b using clean text inputs
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
    Render the matrix input form using clean text inputs (Advanced mode).
    
    Returns:
        Tuple of (A, b, error_message)
    """
    st.header("Enter Matrix A and Vector b")
    st.latex(r"A\hat{x} = \vec{b}")
    
    # Matrix dimension controls
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        num_rows = st.number_input("Rows (m)", min_value=2, max_value=10, value=3, step=1)
    with col2:
        num_cols = st.number_input("Columns (n)", min_value=1, max_value=5, value=2, step=1)
    
    num_rows = int(num_rows)
    num_cols = int(num_cols)
    
    st.markdown("---")
    
    # Default values
    default_A = [[0, 1], [1, 1], [2, 1]]
    default_b = [6, 0, 0]
    
    # Create layout with proper spacing
    col_a, spacer, col_b = st.columns([num_cols * 2, 1, 2])
    
    with col_a:
        st.latex(r"A =")
        
        # Create matrix A input grid
        A_values = []
        for i in range(num_rows):
            cols = st.columns(num_cols)
            row_values = []
            for j in range(num_cols):
                with cols[j]:
                    # Get default value
                    if i < len(default_A) and j < len(default_A[i]):
                        default_val = str(default_A[i][j])
                    else:
                        default_val = "0"
                    
                    val = st.text_input(
                        f"a_{i+1}{j+1}",
                        value=default_val,
                        key=f"A_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    try:
                        row_values.append(float(val) if val else 0.0)
                    except ValueError:
                        row_values.append(0.0)
            A_values.append(row_values)
    
    with col_b:
        st.latex(r"\vec{b} =")
        
        # Create vector b input (single column)
        b_values = []
        for i in range(num_rows):
            # Get default value
            if i < len(default_b):
                default_val = str(default_b[i])
            else:
                default_val = "0"
            
            val = st.text_input(
                f"b_{i+1}",
                value=default_val,
                key=f"b_{i}",
                label_visibility="collapsed"
            )
            try:
                b_values.append(float(val) if val else 0.0)
            except ValueError:
                b_values.append(0.0)
    
    # Convert to numpy arrays
    try:
        A = np.array(A_values, dtype=float)
        b = np.array(b_values, dtype=float)
        
        return A, b, ""
    
    except ValueError as e:
        return None, None, f"Invalid input! Please enter numbers only. Error: {e}"


def parse_data_points(x_input: str, y_input: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Parse x and y input strings into numpy arrays.
    """
    try:
        x_values = np.array([
            float(x.strip()) 
            for x in x_input.replace(',', '\n').split('\n') 
            if x.strip()
        ])
        
        y_values = np.array([
            float(y.strip()) 
            for y in y_input.replace(',', '\n').split('\n') 
            if y.strip()
        ])
        
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
    This function is kept for backwards compatibility.
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
