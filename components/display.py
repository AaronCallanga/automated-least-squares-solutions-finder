"""
Display Components Module

Handles all output rendering for the Least Squares Calculator.
Each function renders a specific section of the step-by-step solution.
Uses LaTeX bmatrix notation for all matrices and vectors.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.least_squares import LeastSquaresResult
from utils.formatting import format_array, format_value


def matrix_to_latex(arr: np.ndarray, use_fractions: bool = False) -> str:
    """Convert a numpy array to LaTeX bmatrix format."""
    if arr.ndim == 1:
        # Vector - single column
        rows = [format_value(v, use_fractions) for v in arr]
        return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"
    else:
        # Matrix - multiple columns
        rows = []
        for row in arr:
            row_str = " & ".join([format_value(v, use_fractions) for v in row])
            rows.append(row_str)
        return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


def render_header():
    """Render the main application header."""
    st.markdown('<h1 class="main-header">Least Squares Calculator</h1>', unsafe_allow_html=True)


def render_introduction():
    """Render the educational introduction section."""
    with st.expander("What is Least Squares? (Click to learn!)", expanded=False):
        st.markdown("""
        ### Understanding Least Squares
        
        The **Least Squares Method** finds the best approximate solution to a system of equations 
        that has no exact solution (overdetermined system).
        
        **When do we use it?**
        - When we have more equations than unknowns
        - When we want to fit a line/curve through data points
        - When we want to minimize the total error
        
        **The Normal Equation:**
        """)
        st.latex(r"A^T A \hat{x} = A^T b")
        st.markdown("""
        **The Solution:**
        """)
        st.latex(r"\hat{x} = (A^T A)^{-1} A^T b")
        st.markdown("""
        **Error (Residual):**
        """)
        st.latex(r"\|\vec{b} - A\hat{x}\|")
        st.markdown("""
        Where:
        - **A** = coefficient matrix (your data)
        - **b** = target/output vector
        - **x̂** = the best solution that minimizes error
        - **Aᵀ** = transpose of A
        """)


def render_step_problem(result: LeastSquaresResult, use_fractions: bool = False):
    """Render Step 1: Problem Setup."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 1: Understanding the Problem")
    st.write("We want to solve **Ax = b** using the Normal Equation:")
    st.latex(r"A^T A \hat{x} = A^T \vec{b}")
    
    st.write("Given:")
    
    # Matrix A
    A_latex = matrix_to_latex(result.A, use_fractions)
    st.latex(r"A = " + A_latex)
    st.caption(f"Shape: {result.A.shape[0]} × {result.A.shape[1]}")
    
    # Vector b
    b_latex = matrix_to_latex(result.b, use_fractions)
    st.latex(r"\vec{b} = " + b_latex)
    st.caption(f"Shape: {len(result.b)} elements")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_transpose(result: LeastSquaresResult, use_fractions: bool = False):
    """Render Step 2: Transpose Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 2: Calculate Aᵀ (Transpose of A)")
    st.write("The transpose flips rows and columns:")
    
    A_latex = matrix_to_latex(result.A, use_fractions)
    AT_latex = matrix_to_latex(result.A_transpose, use_fractions)
    
    st.latex(r"A^T = " + AT_latex)
    st.caption(f"Shape: {result.A.shape} → {result.A_transpose.shape}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_ATA(result: LeastSquaresResult, use_fractions: bool = False):
    """Render Step 3: AᵀA Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 3: Calculate AᵀA (Matrix Multiplication)")
    st.write("Multiply Aᵀ by A:")
    
    AT_latex = matrix_to_latex(result.A_transpose, use_fractions)
    A_latex = matrix_to_latex(result.A, use_fractions)
    ATA_latex = matrix_to_latex(result.ATA, use_fractions)
    
    st.latex(r"A^T A = " + AT_latex + r" \cdot " + A_latex + r" = " + ATA_latex)
    st.caption(f"Shape: {result.ATA.shape[0]} × {result.ATA.shape[1]} (square matrix)")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_inverse(result: LeastSquaresResult, use_fractions: bool = False):
    """Render Step 4: Inverse Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 4: Calculate (AᵀA)⁻¹ (Inverse)")
    st.write("Find the inverse of AᵀA:")
    
    ATA_inv_latex = matrix_to_latex(result.ATA_inverse, use_fractions)
    
    st.latex(r"(A^T A)^{-1} = " + ATA_inv_latex)
    st.success("Matrix is invertible")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_ATb(result: LeastSquaresResult, use_fractions: bool = False):
    """Render Step 5: Aᵀb Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 5: Calculate Aᵀb")
    st.write("Multiply Aᵀ by b:")
    
    AT_latex = matrix_to_latex(result.A_transpose, use_fractions)
    b_latex = matrix_to_latex(result.b, use_fractions)
    ATb_latex = matrix_to_latex(result.ATb, use_fractions)
    
    st.latex(r"A^T \vec{b} = " + AT_latex + r" \cdot " + b_latex + r" = " + ATb_latex)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_step_solution(result: LeastSquaresResult, use_fractions: bool = False):
    """Render Step 6: Final Solution Calculation."""
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("Step 6: Calculate x̂ = (AᵀA)⁻¹Aᵀb")
    st.write("Multiply (AᵀA)⁻¹ by Aᵀb to get our solution:")
    
    ATA_inv_latex = matrix_to_latex(result.ATA_inverse, use_fractions)
    ATb_latex = matrix_to_latex(result.ATb, use_fractions)
    x_hat_latex = matrix_to_latex(result.x_hat, use_fractions)
    
    st.latex(r"\hat{x} = (A^T A)^{-1} A^T \vec{b} = " + ATA_inv_latex + r" \cdot " + ATb_latex + r" = " + x_hat_latex)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_final_result(result: LeastSquaresResult, is_data_points_mode: bool, use_fractions: bool = False):
    """Render the final solution box."""
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.header("Final Solution")
    
    x_hat_latex = matrix_to_latex(result.x_hat, use_fractions)
    st.latex(r"\hat{x} = " + x_hat_latex)
    
    if is_data_points_mode:
        st.markdown("---")
        st.subheader("Best-Fit Line Equation")
        m, c = result.x_hat[0], result.x_hat[1]
        if use_fractions:
            st.latex(f"y = ({format_value(m, True)})x + ({format_value(c, True)})")
        else:
            st.latex(f"y = {m:.4f}x + {c:.4f}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_visualization(result: LeastSquaresResult, is_data_points_mode: bool, x_values: np.ndarray = None):
    """Render the matplotlib visualization with best-fit line and geometric interpretation."""
    st.header("Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and best-fit line (for all modes)
    ax1 = axes[0]
    
    if x_values is not None and len(result.x_hat) >= 2:
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
    elif x_values is not None:
        # Scatter plot only (for single variable)
        ax1.scatter(x_values, result.b, color='#667eea', s=100, zorder=5, 
                   label='Data Points', edgecolors='white', linewidth=2)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('Data Points', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
    else:
        # Fallback: Bar chart for solution values
        ax1.bar(range(len(result.x_hat)), result.x_hat, color='#667eea', 
               edgecolor='white', linewidth=2)
        ax1.set_xlabel('Variable Index', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Solution Values', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(result.x_hat)))
        ax1.set_xticklabels([f'x_{i+1}' for i in range(len(result.x_hat))])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Geometric Interpretation (Column Space Projection)
    ax2 = axes[1]
    
    # For visualization, we'll show the relationship between b, Ax̂ (projection), and error
    # Using first 2-3 components for visualization
    b = result.b
    Ax_hat = result.predicted  # This is Ax̂
    residual = result.residuals  # This is b - Ax̂
    
    n_dim = min(3, len(b))
    
    if n_dim >= 2:
        # 2D visualization of vectors
        ax2.set_xlim(-1, max(abs(b[:2].max()), abs(Ax_hat[:2].max())) * 1.5 + 1)
        ax2.set_ylim(-1, max(abs(b[:2].max()), abs(Ax_hat[:2].max())) * 1.5 + 1)
        
        # Origin
        origin = np.array([0, 0])
        
        # Draw b vector (target)
        ax2.annotate('', xy=(b[0], b[1]), xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='#667eea', lw=2.5))
        ax2.annotate(r'$\vec{b}$', xy=(b[0], b[1]), fontsize=14, color='#667eea',
                    xytext=(5, 5), textcoords='offset points')
        
        # Draw Ax̂ vector (projection onto column space)
        ax2.annotate('', xy=(Ax_hat[0], Ax_hat[1]), xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='#38ef7d', lw=2.5))
        ax2.annotate(r'$A\hat{x}$', xy=(Ax_hat[0], Ax_hat[1]), fontsize=14, color='#38ef7d',
                    xytext=(5, -15), textcoords='offset points')
        
        # Draw error vector (b - Ax̂) - from Ax̂ to b
        ax2.annotate('', xy=(b[0], b[1]), xytext=(Ax_hat[0], Ax_hat[1]),
                    arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2, linestyle='--'))
        mid_x = (b[0] + Ax_hat[0]) / 2
        mid_y = (b[1] + Ax_hat[1]) / 2
        ax2.annotate(r'$\vec{b} - A\hat{x}$', xy=(mid_x, mid_y), fontsize=12, color='#ff6b6b',
                    xytext=(10, 0), textcoords='offset points')
        
        # Draw right angle indicator if error is perpendicular to Ax̂
        # (showing orthogonality)
        scale = 0.3
        if np.linalg.norm(Ax_hat[:2]) > 0.1:
            # Perpendicular indicator
            perp_dir = np.array([-Ax_hat[1], Ax_hat[0]]) / np.linalg.norm(Ax_hat[:2]) * scale
            corner = Ax_hat[:2] - Ax_hat[:2] / np.linalg.norm(Ax_hat[:2]) * scale
            ax2.plot([corner[0], corner[0] + perp_dir[0]], 
                    [corner[1], corner[1] + perp_dir[1]], 
                    color='gray', lw=1)
            ax2.plot([corner[0] + perp_dir[0], Ax_hat[0] + perp_dir[0]], 
                    [corner[1] + perp_dir[1], Ax_hat[1] + perp_dir[1]], 
                    color='gray', lw=1)
        
        # Labels and styling
        ax2.axhline(y=0, color='gray', linestyle='-', lw=0.5)
        ax2.axvline(x=0, color='gray', linestyle='-', lw=0.5)
        ax2.set_xlabel('Component 1', fontsize=12)
        ax2.set_ylabel('Component 2', fontsize=12)
        ax2.set_title('Geometric Interpretation', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal', adjustable='box')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#667eea', lw=2, label=r'$\vec{b}$ (target)'),
            Line2D([0], [0], color='#38ef7d', lw=2, label=r'$A\hat{x}$ (projection)'),
            Line2D([0], [0], color='#ff6b6b', lw=2, linestyle='--', label=r'$\vec{b} - A\hat{x}$ (error)')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Add column space label
        ax2.text(0.95, 0.05, 'Col(A)', transform=ax2.transAxes, fontsize=12,
                fontweight='bold', color='#764ba2', ha='right', va='bottom')
    else:
        # Fallback for 1D
        ax2.bar(['b', 'Ax̂', 'b - Ax̂'], [b[0], Ax_hat[0], residual[0]], 
               color=['#667eea', '#38ef7d', '#ff6b6b'])
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Vector Components', fontsize=14, fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def render_error_analysis(result: LeastSquaresResult, use_fractions: bool = False):
    """Render the error analysis section with vector subtraction and norm."""
    st.subheader("Least-Square Error")
    st.write("Finally, the least-square error is the norm of the vector:")
    
    # Build the LaTeX for vector subtraction: b - Ax̂ = [residuals]
    b_latex = matrix_to_latex(result.b, use_fractions)
    pred_latex = matrix_to_latex(result.predicted, use_fractions)
    res_latex = matrix_to_latex(result.residuals, use_fractions)
    
    # Display vector subtraction
    st.latex(r"\vec{b} - A\hat{x} = " + b_latex + " - " + pred_latex + " = " + res_latex)
    
    st.write("which is")
    
    # Build the norm calculation: ||b - Ax̂|| = √(sum of squares) ≈ result
    squared_terms = [f"({format_value(r, use_fractions)})^2" for r in result.residuals]
    sum_of_squares_latex = " + ".join(squared_terms)
    
    # Calculate the norm
    norm_value = np.sqrt(np.sum(result.residuals ** 2))
    
    st.latex(
        r"\|\vec{b} - A\hat{x}\| = \sqrt{" + sum_of_squares_latex + r"} = \sqrt{" + 
        format_value(result.sse, use_fractions) + r"} \approx " + f"{norm_value:.4f}"
    )
    
    st.info("Lower error values indicate a better fit. The least squares method minimizes ‖b - Ax̂‖².")


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Least Squares Calculator | Built with Streamlit, NumPy & Matplotlib</p>
        <p>Perfect for learning linear algebra</p>
    </div>
    """, unsafe_allow_html=True)
