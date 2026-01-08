"""
Display Components Module

Handles all output rendering for the Least Squares Calculator.
Each function renders a specific section of the step-by-step solution.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.least_squares import LeastSquaresResult


def render_header():
    """Render the main application header."""
    st.markdown('<h1 class="main-header">📐 Least Squares Calculator</h1>', unsafe_allow_html=True)


def render_introduction():
    """Render the educational introduction section."""
    with st.expander("📚 What is Least Squares? (Click to learn!)", expanded=False):
        st.markdown("""
        ### Understanding Least Squares
        
        The **Least Squares Method** finds the best approximate solution to a system of equations 
        that has no exact solution (overdetermined system).
        
        **When do we use it?**
        - When we have more equations than unknowns
        - When we want to fit a line/curve through data points
        - When we want to minimize the total error
        
        **The Formula:**
        """)
        st.latex(r"\hat{x} = (A^T A)^{-1} A^T b")
        st.markdown("""
        Where:
        - **A** = coefficient matrix (your data)
        - **b** = target/output vector
        - **x̂** = the best solution that minimizes error
        - **Aᵀ** = transpose of A
        """)


def render_step_problem(result: LeastSquaresResult):
    """Render Step 1: Problem Setup."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 1️⃣: Understanding the Problem")
    st.write("We want to solve **Ax = b** using the least squares formula:")
    st.latex(r"\hat{x} = (A^T A)^{-1} A^T b")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Matrix A:**")
        st.write(result.A)
        st.write(f"Shape: {result.A.shape[0]} rows × {result.A.shape[1]} columns")
    with col2:
        st.write("**Vector b:**")
        st.write(result.b)
        st.write(f"Shape: {len(result.b)} elements")
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_transpose(result: LeastSquaresResult):
    """Render Step 2: Transpose Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 2️⃣: Calculate Aᵀ (Transpose of A)")
    st.write("The transpose flips rows and columns:")
    st.latex(r"A^T = \text{swap rows and columns of } A")
    st.write("**Aᵀ =**")
    st.write(result.A_transpose)
    st.write(f"Shape changed from {result.A.shape} → {result.A_transpose.shape}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_ATA(result: LeastSquaresResult):
    """Render Step 3: AᵀA Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 3️⃣: Calculate AᵀA (Matrix Multiplication)")
    st.write("Multiply Aᵀ by A:")
    st.latex(r"A^T A = A^T \times A")
    st.write("**AᵀA =**")
    st.write(result.ATA)
    st.write(f"Result shape: {result.ATA.shape[0]} × {result.ATA.shape[1]} (square matrix)")
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_inverse(result: LeastSquaresResult):
    """Render Step 4: Inverse Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 4️⃣: Calculate (AᵀA)⁻¹ (Inverse)")
    st.write("Find the inverse of AᵀA:")
    st.latex(r"(A^T A)^{-1}")
    st.write("**(AᵀA)⁻¹ =**")
    st.write(result.ATA_inverse)
    st.success("✅ Matrix is invertible!")
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_ATb(result: LeastSquaresResult):
    """Render Step 5: Aᵀb Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 5️⃣: Calculate Aᵀb")
    st.write("Multiply Aᵀ by b:")
    st.latex(r"A^T b = A^T \times b")
    st.write("**Aᵀb =**")
    st.write(result.ATb)
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_solution(result: LeastSquaresResult):
    """Render Step 6: Final Solution Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 6️⃣: Calculate x̂ = (AᵀA)⁻¹Aᵀb")
    st.write("Multiply (AᵀA)⁻¹ by Aᵀb to get our solution:")
    st.latex(r"\hat{x} = (A^T A)^{-1} A^T b")
    st.markdown('</div>', unsafe_allow_html=True)


def render_final_result(result: LeastSquaresResult, is_data_points_mode: bool):
    """Render the final solution box."""
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.header("🎯 Final Solution")
    st.write("**x̂ =**")
    
    for i, val in enumerate(result.x_hat):
        st.write(f"x_{i+1} = {val:.6f}")
    
    if is_data_points_mode:
        st.markdown("---")
        st.subheader("📈 Best-Fit Line Equation:")
        m, c = result.x_hat[0], result.x_hat[1]
        st.latex(f"y = {m:.4f}x + {c:.4f}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_visualization(result: LeastSquaresResult, is_data_points_mode: bool, x_values: np.ndarray = None):
    """Render the matplotlib visualization."""
    st.header("📊 Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and best-fit line
    ax1 = axes[0]
    if is_data_points_mode and x_values is not None:
        # Scatter plot with best-fit line
        ax1.scatter(x_values, result.b, color='#667eea', s=100, zorder=5, 
                   label='Data Points', edgecolors='white', linewidth=2)
        x_line = np.linspace(min(x_values) - 0.5, max(x_values) + 0.5, 100)
        y_line = result.x_hat[0] * x_line + result.x_hat[1]
        ax1.plot(x_line, y_line, color='#38ef7d', linewidth=3, 
                label=f'Best Fit: y = {result.x_hat[0]:.3f}x + {result.x_hat[1]:.3f}')
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('Data Points & Best-Fit Line', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
    else:
        # Bar chart for matrix mode
        ax1.bar(range(len(result.x_hat)), result.x_hat, color='#667eea', 
               edgecolor='white', linewidth=2)
        ax1.set_xlabel('Variable Index', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Solution Values (x̂)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(result.x_hat)))
        ax1.set_xticklabels([f'x_{i+1}' for i in range(len(result.x_hat))])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    if is_data_points_mode and x_values is not None:
        ax2.stem(x_values, result.residuals, linefmt='#764ba2', markerfmt='o', basefmt='gray')
        ax2.set_xlabel('X', fontsize=12)
    else:
        ax2.stem(range(len(result.residuals)), result.residuals, 
                linefmt='#764ba2', markerfmt='o', basefmt='gray')
        ax2.set_xlabel('Observation Index', fontsize=12)
    
    ax2.axhline(y=0, color='#38ef7d', linestyle='--', linewidth=2)
    ax2.set_ylabel('Residual (Error)', fontsize=12)
    ax2.set_title('Residuals (b - Ax̂)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def render_error_analysis(result: LeastSquaresResult):
    """Render the error analysis section."""
    st.subheader("📉 Error Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sum of Squared Errors (SSE)", f"{result.sse:.6f}")
    with col2:
        st.metric("Mean Squared Error (MSE)", f"{result.mse:.6f}")
    with col3:
        st.metric("Root Mean Squared Error (RMSE)", f"{result.rmse:.6f}")
    
    st.info("💡 **Lower error values indicate a better fit!** The least squares method minimizes the Sum of Squared Errors (SSE).")


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>📐 Least Squares Calculator | Built with Streamlit, NumPy & Matplotlib</p>
        <p>Perfect for learning linear algebra! 🎓</p>
    </div>
    """, unsafe_allow_html=True)
