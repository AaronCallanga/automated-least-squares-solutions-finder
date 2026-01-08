"""
Display Components Module

Handles all output rendering for the Least Squares Calculator.
Each function renders a specific section of the step-by-step solution.
Uses LaTeX bmatrix notation for all matrices and vectors.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
        st.markdown("### The Problem")
        st.markdown("""
        Sometimes we have a system of equations that **has no exact solution**. 
        This happens when we have more equations than unknowns (for example, trying to fit a line through many data points).
        
        In these cases, we can't find a perfect answer, but we can find the **best approximate answer** — 
        one that gets as close as possible to all our data points.
        """)
        
        st.markdown("### What is Least Squares?")
        st.markdown("""
        The **Least Squares Method** finds the best approximate solution by **minimizing the total error**.
        
        Think of it this way: if we can't hit the target exactly, we want to get as close as possible. 
        The "least squares" solution is the one where the sum of all squared errors is the smallest.
        """)
        
        st.markdown("### The Math (Made Simple)")
        st.markdown("We start with a system of equations:")
        st.latex(r"A\vec{x} = \vec{b}")
        
        st.markdown("""
        Where:
        - **A** is a table of numbers (matrix) representing our data
        - **x** is what we're trying to find
        - **b** is our target values
        """)
        
        st.markdown("Since we can't solve this exactly, we look for the best estimate:")
        st.latex(r"\hat{x}")
        st.markdown("(read as 'x-hat')")
        
        st.markdown("---")
        st.markdown("### The Error")
        st.markdown("The error is the difference between our target and what we get:")
        st.latex(r"\vec{b} - A\hat{x} = \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_n \end{bmatrix}")
        
        st.markdown("The **least-squares error** is:")
        st.latex(r"\|\vec{b} - A\hat{x}\| = \sqrt{e_1^2 + e_2^2 + \cdots + e_n^2}")
        
        st.markdown("""
        We call it "least squares" because we're **minimizing the sum of squared errors**.
        """)
        
        st.markdown("---")
        st.markdown("### The Solution")
        st.markdown("To find the best answer, we solve what's called the **Normal Equation**:")
        st.latex(r"A^T A \hat{x} = A^T \vec{b}")
        
        st.markdown("Which gives us the formula:")
        st.latex(r"\hat{x} = (A^T A)^{-1} A^T \vec{b}")
        
        st.markdown("""
        Where:
        - **Aᵀ** is A "flipped" (transposed) — rows become columns
        - **(AᵀA)⁻¹** is the inverse (like dividing by AᵀA)
        """)
        
        st.markdown("""
        > **Remark:** If A has a full column rank, then AᵀA is nonsingular (invertible) 
        > and the least-squares solution is **unique**.
        """)
        
        st.markdown("---")
        st.markdown("### Real-World Example")
        st.markdown("""
        **Fitting a line through data points:**
        
        If you have several (x, y) points and want to draw the best straight line through them, 
        you're solving a least squares problem! The method finds the line that minimizes 
        the total distance from each point to the line.
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
    """Render visualizations: Matplotlib for best-fit line, Plotly for 3D geometric interpretation."""
    st.header("Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Best-Fit Line / Solution", "3D Geometric Interpretation"])
    
    with tab1:
        # Matplotlib visualization for data points and best-fit line
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if x_values is not None and len(result.x_hat) >= 2:
            # Scatter plot with best-fit line
            ax.scatter(x_values, result.b, color='#667eea', s=100, zorder=5, 
                       label='Data Points', edgecolors='white', linewidth=2)
            x_line = np.linspace(min(x_values) - 0.5, max(x_values) + 0.5, 100)
            y_line = result.x_hat[0] * x_line + result.x_hat[1]
            ax.plot(x_line, y_line, color='#38ef7d', linewidth=3, 
                    label=f'Best Fit: y = {result.x_hat[0]:.3f}x + {result.x_hat[1]:.3f}')
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_title('Data Points & Best-Fit Line', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
        elif x_values is not None:
            ax.scatter(x_values, result.b, color='#667eea', s=100, zorder=5, 
                       label='Data Points', edgecolors='white', linewidth=2)
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_title('Data Points', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
        else:
            ax.bar(range(len(result.x_hat)), result.x_hat, color='#667eea', 
                   edgecolor='white', linewidth=2)
            ax.set_xlabel('Variable Index', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Solution Values', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(result.x_hat)))
            ax.set_xticklabels([f'x_{i+1}' for i in range(len(result.x_hat))])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # 3D Plotly visualization for geometric interpretation
        render_3d_geometric_visualization(result)


def render_3d_geometric_visualization(result: LeastSquaresResult):
    """Render interactive 3D visualization of the column space projection."""
    
    b = result.b
    Ax_hat = result.predicted
    A = result.A
    
    # Pad vectors to 3D if needed
    def to_3d(v):
        if len(v) >= 3:
            return v[:3]
        elif len(v) == 2:
            return np.array([v[0], v[1], 0])
        else:
            return np.array([v[0], 0, 0])
    
    b_3d = to_3d(b)
    Ax_hat_3d = to_3d(Ax_hat)
    
    # Get column vectors of A (for spanning the plane)
    if A.shape[1] >= 2:
        v1 = to_3d(A[:, 0])
        v2 = to_3d(A[:, 1])
    elif A.shape[1] == 1:
        v1 = to_3d(A[:, 0])
        v2 = np.array([0, 0, 1])  # Default second vector
    else:
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
    
    # Create the figure
    fig = go.Figure()
    
    # Calculate range for the plane
    max_range = max(
        np.max(np.abs(b_3d)) if len(b_3d) > 0 else 1,
        np.max(np.abs(Ax_hat_3d)) if len(Ax_hat_3d) > 0 else 1,
        np.max(np.abs(v1)) if len(v1) > 0 else 1,
        np.max(np.abs(v2)) if len(v2) > 0 else 1
    ) * 1.5
    
    # Create column space plane (Col A)
    # Generate a mesh for the plane spanned by v1 and v2
    s = np.linspace(-max_range, max_range, 10)
    t = np.linspace(-max_range, max_range, 10)
    S, T = np.meshgrid(s, t)
    
    # Plane: P = s*v1 + t*v2
    X_plane = S * v1[0] + T * v2[0]
    Y_plane = S * v1[1] + T * v2[1]
    Z_plane = S * v1[2] + T * v2[2]
    
    # Add the column space plane (semi-transparent purple)
    fig.add_trace(go.Surface(
        x=X_plane, y=Y_plane, z=Z_plane,
        colorscale=[[0, 'rgba(180, 120, 200, 0.3)'], [1, 'rgba(180, 120, 200, 0.3)']],
        showscale=False,
        name='Col(A)',
        hoverinfo='name'
    ))
    
    # Add column vectors v1 and v2 (orange)
    fig.add_trace(go.Scatter3d(
        x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
        mode='lines+markers',
        marker=dict(size=5, color='#ff7f0e'),
        line=dict(color='#ff7f0e', width=6),
        name='v₁ (column 1)'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]],
        mode='lines+markers',
        marker=dict(size=5, color='#ff7f0e'),
        line=dict(color='#ff7f0e', width=6),
        name='v₂ (column 2)'
    ))
    
    # Add target vector b (red)
    fig.add_trace(go.Scatter3d(
        x=[0, b_3d[0]], y=[0, b_3d[1]], z=[0, b_3d[2]],
        mode='lines+markers',
        marker=dict(size=8, color='#d62728', symbol='diamond'),
        line=dict(color='#d62728', width=6),
        name='b (target)'
    ))
    
    # Add b point marker
    fig.add_trace(go.Scatter3d(
        x=[b_3d[0]], y=[b_3d[1]], z=[b_3d[2]],
        mode='markers+text',
        marker=dict(size=10, color='#d62728'),
        text=['b'],
        textposition='top center',
        textfont=dict(size=14, color='#d62728'),
        name='b point',
        showlegend=False
    ))
    
    # Add projection Ax̂ (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, Ax_hat_3d[0]], y=[0, Ax_hat_3d[1]], z=[0, Ax_hat_3d[2]],
        mode='lines+markers',
        marker=dict(size=8, color='#1f77b4'),
        line=dict(color='#1f77b4', width=6),
        name='Ax̂ (projection)'
    ))
    
    # Add Ax̂ point marker
    fig.add_trace(go.Scatter3d(
        x=[Ax_hat_3d[0]], y=[Ax_hat_3d[1]], z=[Ax_hat_3d[2]],
        mode='markers+text',
        marker=dict(size=10, color='#1f77b4'),
        text=['Ax̂'],
        textposition='bottom center',
        textfont=dict(size=14, color='#1f77b4'),
        name='Ax̂ point',
        showlegend=False
    ))
    
    # Add error vector (dashed line from Ax̂ to b)
    fig.add_trace(go.Scatter3d(
        x=[Ax_hat_3d[0], b_3d[0]], y=[Ax_hat_3d[1], b_3d[1]], z=[Ax_hat_3d[2], b_3d[2]],
        mode='lines',
        line=dict(color='#ff6b6b', width=4, dash='dash'),
        name='b - Ax̂ (error)'
    ))
    
    # Add origin marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=6, color='black'),
        text=['O'],
        textposition='bottom left',
        textfont=dict(size=12),
        name='Origin',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        title=dict(
            text='Geometric Interpretation: Column Space Projection',
            font=dict(size=16)
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add annotation explaining the visualization
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **How to read this visualization:**
    - **Purple plane** = Column space Col(A) - all possible outputs of A
    - **Orange vectors** (v₁, v₂) = Columns of matrix A
    - **Red point** (b) = Target vector (what we want)
    - **Blue point** (Ax̂) = Best approximation (projection onto Col A)
    - **Dashed line** = Error vector (perpendicular to the plane!)
    
    **Drag to rotate, scroll to zoom!**
    """)


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
