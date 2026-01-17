"""
Input Components Module

Handles all user input forms and parsing for the Least Squares Calculator.
Provides two input modes:
    1. Data Points (Easy) - Enter x,y coordinates for line fitting
    2. Matrix Input (Advanced) - Enter matrix A and vector b using clean text inputs
"""
import streamlit as st
import numpy as np
import re
from fractions import Fraction
from typing import Tuple, Optional, List


def parse_fraction(value_str: str) -> float:
    """
    Parse a string that could be a fraction (e.g., '1/2') or decimal.
    
    Args:
        value_str: String like '1/2', '3/4', '2.5', or '3'
    
    Returns:
        Float value
    """
    value_str = value_str.strip()
    if not value_str:
        return 0.0
    
    try:
        if '/' in value_str:
            return float(Fraction(value_str))
        else:
            return float(value_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


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
    
    # Matrix dimension controls - centered
    left_spacer, center_col, right_spacer = st.columns([2, 3, 2])
    with center_col:
        st.markdown("**Matrix Dimensions**")
        dim_col1, dim_col2 = st.columns(2)
        with dim_col1:
            num_rows = st.number_input("Rows (m)", min_value=2, max_value=10, value=3, step=1)
        with dim_col2:
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
                    row_values.append(parse_fraction(val))
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
            b_values.append(parse_fraction(val))
    
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


def render_linear_system_input() -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Render the linear system input form where users enter equations directly.
    
    Example equations:
        x1 + 2x2 = 4
        2x1 - x2 = 3
        x1 + 3x2 = 6
    
    Returns:
        Tuple of (A, b, error_message)
    """
    st.header("Enter System of Linear Equations")
    st.latex(r"A\vec{x} = \vec{b}")
    
    st.info("""
    **How to enter equations:**
    - Use any letters for variables: `a`, `b`, `c` or `x`, `y`, `z` or `x1`, `x2`, `x3`
    - Use `+` and `-` for operations
    - Each equation on a new line
    - Examples: `a + 2b = 4`, `2x - 3y + z = 5`, `x1 + x2 = 10`
    """)
    
    # Default example equations
    default_equations = """a + 2b = 4
2a - b = 3
a + 3b = 6"""
    
    equations_input = st.text_area(
        "Enter your equations (one per line):",
        value=default_equations,
        height=150,
        help="Enter linear equations with variables x1, x2, x3, etc."
    )
    
    # Parse the equations
    try:
        A, b, var_names = parse_linear_system(equations_input)
        
        if A is not None and len(var_names) > 0:
            # Show extracted matrices using LaTeX
            st.markdown("---")
            st.markdown("**Extracted from equations:**")
            
            # Build LaTeX for matrix A
            A_rows = []
            for row in A:
                A_rows.append(" & ".join([str(int(v)) if v == int(v) else f"{v:.2f}" for v in row]))
            A_latex = r"\begin{bmatrix}" + r"\\".join(A_rows) + r"\end{bmatrix}"
            
            # Build LaTeX for vector x (variable names)
            x_latex = r"\begin{bmatrix}" + r"\\".join(var_names) + r"\end{bmatrix}"
            
            # Build LaTeX for vector b
            b_latex = r"\begin{bmatrix}" + r"\\".join([str(int(v)) if v == int(v) else f"{v:.2f}" for v in b]) + r"\end{bmatrix}"
            
            # Display all inline
            st.latex(r"A = " + A_latex)
            st.latex(r"\vec{x} = " + x_latex)
            st.latex(r"\vec{b} = " + b_latex)
            
            return A, b, ""
        else:
            return None, None, "Could not parse equations"
    
    except Exception as e:
        return None, None, f"Error parsing equations: {e}"


def parse_linear_system(equations_str: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Parse a string of linear equations into matrix A and vector b.
    Supports flexible variable names: a, b, c or x, y, z or x1, x2, x3
    
    Args:
        equations_str: String containing linear equations, one per line
    
    Returns:
        Tuple of (A, b, variable_names)
    """
    lines = [line.strip() for line in equations_str.strip().split('\n') if line.strip()]
    
    if not lines:
        return None, None, []
    
    # Find all variables used (letters or letter+number combinations)
    all_vars = set()
    for line in lines:
        # Match patterns: single letters (a, b, x, y) or letter+number (x1, x2)
        left_side = line.split('=')[0] if '=' in line else line
        # Find all variable patterns
        vars_found = re.findall(r'[a-zA-Z]\d*', left_side.replace(' ', ''))
        # Filter out just coefficient-looking things
        for v in vars_found:
            if v and not v.replace('.', '').isdigit():
                all_vars.add(v.lower())
    
    if not all_vars:
        return None, None, []
    
    # Sort variables naturally
    var_list = sorted(list(all_vars), key=lambda x: (len(x), x))
    var_to_idx = {v: i for i, v in enumerate(var_list)}
    
    num_vars = len(var_list)
    num_equations = len(lines)
    
    A = np.zeros((num_equations, num_vars))
    b = np.zeros(num_equations)
    
    for i, line in enumerate(lines):
        # Split by '='
        if '=' not in line:
            continue
        
        left_side, right_side = line.split('=')
        
        # Parse right side (b value)
        b[i] = parse_fraction(right_side.strip())
        
        # Parse left side (coefficients)
        left_side = left_side.replace(' ', '').replace('-', '+-')
        terms = [t for t in left_side.split('+') if t]
        
        for term in terms:
            # Match patterns like: 2a, -3b, a, -x, 2x1, x2
            match = re.match(r'([+-]?[\d./]*)([a-zA-Z]\d*)', term)
            if match:
                coef_str = match.group(1)
                var_name = match.group(2).lower()
                
                if coef_str in ['', '+']:
                    coef = 1.0
                elif coef_str == '-':
                    coef = -1.0
                else:
                    coef = parse_fraction(coef_str)
                
                if var_name in var_to_idx:
                    A[i, var_to_idx[var_name]] = coef
    
    return A, b, var_list

