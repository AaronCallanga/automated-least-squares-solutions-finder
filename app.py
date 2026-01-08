"""
Least Squares Calculator - Main Application

A beginner-friendly calculator that demonstrates the least squares method
step-by-step, using the Normal Equation: AᵀAx̂ = Aᵀb

Application Flow:
    1. User selects input mode (Data Points or Matrix)
    2. User selects display format (Decimal or Fraction)
    3. User enters their data
    4. User clicks Calculate
    5. App shows step-by-step solution
    6. App displays visualization and error analysis

Project Structure:
    app.py              - Main application flow (this file)
    styles/css.py       - Custom CSS styles
    utils/least_squares.py - Core calculation logic
    utils/formatting.py    - Number formatting (decimal/fraction)
    components/inputs.py   - Input form components
    components/display.py  - Output display components
"""
import streamlit as st
import numpy as np

# Import our modules
from styles import apply_custom_styles
from utils import LeastSquaresSolver
from utils.least_squares import create_design_matrix
from components import (
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
    render_final_result,
    render_visualization,
    render_error_analysis,
    render_footer
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Least Squares Calculator",
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
st.sidebar.header("⚙️ Settings")

# Input mode selection
input_mode = st.sidebar.radio(
    "Choose input method:",
    ["📊 Data Points (Easy)", "🔢 Matrix Input (Advanced)"]
)
is_data_points_mode = input_mode == "📊 Data Points (Easy)"

st.sidebar.markdown("---")

# Display format toggle
st.sidebar.subheader("📐 Display Format")
use_fractions = st.sidebar.toggle(
    "Show as Fractions",
    value=False,
    help="Toggle between decimal and fraction display for matrices and results"
)

if use_fractions:
    st.sidebar.info("📊 Displaying values as **fractions**")
else:
    st.sidebar.info("📊 Displaying values as **decimals**")


# =============================================================================
# USER INPUT SECTION
# =============================================================================
x_values = None  # Only used in data points mode

if is_data_points_mode:
    # Easy mode: Enter x,y coordinates
    x_input, y_input = render_data_points_input()
    x_values, y_values, error_msg = parse_data_points(x_input, y_input)
    
    if error_msg:
        st.error(f"❌ {error_msg}")
        st.stop()
    
    # Create design matrix for line fitting (y = mx + c)
    A = create_design_matrix(x_values)
    b = y_values
else:
    # Advanced mode: Enter matrix A and vector b directly
    a_input, b_input = render_matrix_input()
    A, b, error_msg = parse_matrix_input(a_input, b_input)
    
    if error_msg:
        st.error(f"❌ {error_msg}")
        st.stop()


# =============================================================================
# CALCULATE BUTTON
# =============================================================================
st.markdown("---")
calculate = st.button("🚀 Calculate Least Squares Solution", type="primary", use_container_width=True)


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
    
    # Display step-by-step solution
    st.markdown("---")
    st.header("📝 Step-by-Step Solution")
    
    render_step_problem(result, use_fractions)      # Step 1: Show A and b
    render_step_transpose(result, use_fractions)    # Step 2: Calculate Aᵀ
    render_step_ATA(result, use_fractions)          # Step 3: Calculate AᵀA
    render_step_inverse(result, use_fractions)      # Step 4: Calculate (AᵀA)⁻¹
    render_step_ATb(result, use_fractions)          # Step 5: Calculate Aᵀb
    render_step_solution(result, use_fractions)     # Step 6: Calculate x̂
    
    # Display final result
    st.markdown("---")
    render_final_result(result, is_data_points_mode, use_fractions)
    
    # Display visualization
    st.markdown("---")
    render_visualization(result, is_data_points_mode, x_values)
    
    # Display error analysis
    render_error_analysis(result, use_fractions)


# =============================================================================
# FOOTER
# =============================================================================
render_footer()
