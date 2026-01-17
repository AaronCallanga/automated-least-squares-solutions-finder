import streamlit as st
import numpy as np

# Import our modules from src/
from src.styles import apply_custom_styles
from src.utils import LeastSquaresSolver
from src.utils.least_squares import create_design_matrix
from src.components import (
    # Input components
    render_data_points_input,
    render_matrix_input,
    parse_data_points,
    parse_matrix_input,
    # Display components
    render_header,
    render_introduction,
    render_step_problem,
    render_step_transpose,
    render_step_ATA,
    render_step_inverse,
    render_step_ATb,
    render_step_solution,
    render_quick_answer,
    render_final_result,
    render_visualization,
    render_error_analysis,
    render_footer
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Automated Least Squares Solutions Finder",
    page_icon="📐",
    layout="wide"
)


# =============================================================================
# APPLY STYLES
# =============================================================================
apply_custom_styles()


# =============================================================================
# HEADER & INTRODUCTION
# =============================================================================
render_header()
render_introduction()
st.markdown("---")


# =============================================================================
# SIDEBAR - SETTINGS
# =============================================================================
st.sidebar.header("Settings")

# Input mode selection
input_mode = st.sidebar.radio(
    "Choose input method:",
    ["Data Points", "Linear System", "Matrix Input"]
)
is_data_points_mode = input_mode == "Data Points"
is_linear_system_mode = input_mode == "Linear System"

st.sidebar.markdown("---")

# Display format toggle
st.sidebar.subheader("Display Format")
use_fractions = st.sidebar.toggle(
    "Show as Fractions",
    value=True,
    help="Toggle between decimal and fraction display for matrices and results"
)

if use_fractions:
    st.sidebar.info("Displaying values as **fractions**")
else:
    st.sidebar.info("Displaying values as **decimals**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Group 1 | Automated Least Squares Solutions Finder")
st.sidebar.markdown("""
- Mark Jason Manlapaz
- Jomar Escala 
- Christine Rio 
- Anthony Bonito 
- Aaron Dave Callanga
""")


# =============================================================================
# USER INPUT SECTION
# =============================================================================
x_values = None  # Only used in data points mode

if is_data_points_mode:
    # Easy mode: Enter x,y coordinates
    x_input, y_input = render_data_points_input()
    x_values, y_values, error_msg = parse_data_points(x_input, y_input)
    
    if error_msg:
        st.error(f"{error_msg}")
        st.stop()
    
    # Create design matrix for line fitting (y = mx + c)
    A = create_design_matrix(x_values)
    b = y_values

elif is_linear_system_mode:
    # Linear system mode: Enter equations directly
    from components import render_linear_system_input
    A, b, error_msg = render_linear_system_input()
    
    if error_msg:
        st.error(f"{error_msg}")
        st.stop()
    
    # For linear system mode, use first column as x values for visualization
    if A is not None and A.shape[1] >= 1:
        x_values = A[:, 0]

else:
    # Advanced mode: Enter matrix A and vector b using editable tables
    A, b, error_msg = render_matrix_input()
    
    if error_msg:
        st.error(f"{error_msg}")
        st.stop()
    
    # For matrix mode, use first column as x values for visualization
    if A is not None and A.shape[1] >= 1:
        x_values = A[:, 0]


# =============================================================================
# CALCULATE BUTTON
# =============================================================================
st.markdown("---")
calculate = st.button("Calculate Least Squares Solution", type="primary", use_container_width=True)


# =============================================================================
# CALCULATION & RESULTS
# =============================================================================
if calculate:
    # Initialize solver and compute solution
    solver = LeastSquaresSolver()
    result = solver.solve(A, b)
    
    # Check for errors
    if not result.is_valid:
        st.error(f"❌ {result.error_message}")
        st.stop()
    
    # Display quick answer first
    st.markdown("---")
    render_quick_answer(result, is_data_points_mode, use_fractions)
    
    # Display visualization right after answer
    st.markdown("---")
    render_visualization(result, is_data_points_mode, x_values)
    
    # Display step-by-step solution
    st.markdown("---")
    st.header("Step-by-Step Solution")
    
    with st.expander("Step 1: Problem Setup (Matrix A and Vector b)", expanded=False):
        render_step_problem(result, use_fractions)
    
    with st.expander("Step 2: Calculate Aᵀ (Transpose)", expanded=False):
        render_step_transpose(result, use_fractions)
    
    with st.expander("Step 3: Calculate AᵀA", expanded=False):
        render_step_ATA(result, use_fractions)
    
    with st.expander("Step 4: Calculate (AᵀA)⁻¹ (Inverse)", expanded=False):
        render_step_inverse(result, use_fractions)
    
    with st.expander("Step 5: Calculate Aᵀb", expanded=False):
        render_step_ATb(result, use_fractions)
    
    with st.expander("Step 6: Calculate x̂ = (AᵀA)⁻¹Aᵀb", expanded=False):
        render_step_solution(result, use_fractions)
    
    with st.expander("Step 7: Final Solution", expanded=False):
        render_final_result(result, is_data_points_mode, use_fractions)
    
    with st.expander("Step 8: Least-Squares Error Calculation", expanded=False):
        render_error_analysis(result, use_fractions)


# =============================================================================
# FOOTER
# =============================================================================
render_footer()
