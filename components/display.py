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
    x_hat = result.x_hat
    
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    
    # For higher dimensional data, show a warning
    if num_rows > 3:
        st.warning(f"""
        **Note:** Your system has {num_rows} dimensions, but we can only visualize 3 dimensions.  
        This 3D plot shows the **first 3 components** of each vector.  
        The mathematical calculations are still done in full {num_rows}D space!
        """)
    
    # Pad or truncate vectors to 3D
    def to_3d(v):
        v = np.asarray(v, dtype=float)
        if len(v) >= 3:
            return v[:3]
        elif len(v) == 2:
            return np.array([v[0], v[1], 0.0])
        elif len(v) == 1:
            return np.array([v[0], 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])
    
    b_3d = to_3d(b)
    Ax_hat_3d = to_3d(Ax_hat)
    
    # Get column vectors of A (in 3D)
    column_vectors = [to_3d(A[:, i]) for i in range(num_cols)]
    
    # Colors for column vectors
    col_colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
    
    # Create the figure
    fig = go.Figure()
    
    # Calculate range for visualization
    all_vectors = [b_3d, Ax_hat_3d] + column_vectors
    max_val = max([np.max(np.abs(v)) for v in all_vectors] + [1])
    max_range = max_val * 1.3
    
    # === Column Space Plane (Col A) ===
    if num_cols >= 2:
        v1, v2 = column_vectors[0], column_vectors[1]
        
        # Create a finer mesh for the plane
        s = np.linspace(-max_range, max_range, 25)
        t = np.linspace(-max_range, max_range, 25)
        S, T = np.meshgrid(s, t)
        
        X_plane = S * v1[0] + T * v2[0]
        Y_plane = S * v1[1] + T * v2[1]
        Z_plane = S * v1[2] + T * v2[2]
        
        # Purple semi-transparent plane
        fig.add_trace(go.Surface(
            x=X_plane, y=Y_plane, z=Z_plane,
            colorscale=[[0, 'rgba(160, 100, 200, 0.35)'], [1, 'rgba(180, 120, 220, 0.35)']],
            showscale=False,
            name='Col(A) - Column Space',
            hoverinfo='name',
            opacity=0.5,
            contours=dict(
                x=dict(show=True, color='rgba(150,100,200,0.2)', width=1),
                y=dict(show=True, color='rgba(150,100,200,0.2)', width=1)
            )
        ))
    elif num_cols == 1:
        v1 = column_vectors[0]
        t_vals = np.linspace(-max_range, max_range, 100)
        fig.add_trace(go.Scatter3d(
            x=t_vals * v1[0], y=t_vals * v1[1], z=t_vals * v1[2],
            mode='lines',
            line=dict(color='rgba(160, 100, 200, 0.6)', width=10),
            name='Col(A) - Column Space (line)'
        ))
    
    # === Column vectors with arrowheads ===
    for i, v in enumerate(column_vectors):
        color = col_colors[i % len(col_colors)]
        v_len = np.linalg.norm(v)
        
        if v_len > 0.01:
            # Vector line
            fig.add_trace(go.Scatter3d(
                x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
                mode='lines',
                line=dict(color=color, width=6),
                name=f'v{i+1} (column {i+1})'
            ))
            
            # Arrowhead cone
            v_norm = v / v_len
            arrow_size = max_range * 0.06
            fig.add_trace(go.Cone(
                x=[v[0]], y=[v[1]], z=[v[2]],
                u=[v_norm[0]], v=[v_norm[1]], w=[v_norm[2]],
                sizemode='absolute', sizeref=arrow_size,
                colorscale=[[0, color], [1, color]],
                showscale=False, showlegend=False, hoverinfo='skip'
            ))
    
    # === Target vector b (red) ===
    b_len = np.linalg.norm(b_3d)
    if b_len > 0.01:
        fig.add_trace(go.Scatter3d(
            x=[0, b_3d[0]], y=[0, b_3d[1]], z=[0, b_3d[2]],
            mode='lines',
            line=dict(color='#d62728', width=6),
            name='b (target vector)'
        ))
        b_norm = b_3d / b_len
        fig.add_trace(go.Cone(
            x=[b_3d[0]], y=[b_3d[1]], z=[b_3d[2]],
            u=[b_norm[0]], v=[b_norm[1]], w=[b_norm[2]],
            sizemode='absolute', sizeref=max_range * 0.06,
            colorscale=[[0, '#d62728'], [1, '#d62728']],
            showscale=False, showlegend=False, hoverinfo='skip'
        ))
    
    # b label
    fig.add_trace(go.Scatter3d(
        x=[b_3d[0]], y=[b_3d[1]], z=[b_3d[2]],
        mode='markers+text',
        marker=dict(size=10, color='#d62728', symbol='diamond'),
        text=['b'], textposition='top right',
        textfont=dict(size=18, color='#d62728', family='Arial Black'),
        showlegend=False, hoverinfo='text', hovertext='Target b'
    ))
    
    # === Projection Ax̂ (blue) ===
    Ax_len = np.linalg.norm(Ax_hat_3d)
    if Ax_len > 0.01:
        fig.add_trace(go.Scatter3d(
            x=[0, Ax_hat_3d[0]], y=[0, Ax_hat_3d[1]], z=[0, Ax_hat_3d[2]],
            mode='lines',
            line=dict(color='#1f77b4', width=6),
            name='Ax̂ (projection onto Col A)'
        ))
        Ax_norm = Ax_hat_3d / Ax_len
        fig.add_trace(go.Cone(
            x=[Ax_hat_3d[0]], y=[Ax_hat_3d[1]], z=[Ax_hat_3d[2]],
            u=[Ax_norm[0]], v=[Ax_norm[1]], w=[Ax_norm[2]],
            sizemode='absolute', sizeref=max_range * 0.06,
            colorscale=[[0, '#1f77b4'], [1, '#1f77b4']],
            showscale=False, showlegend=False, hoverinfo='skip'
        ))
    
    # Ax̂ label
    fig.add_trace(go.Scatter3d(
        x=[Ax_hat_3d[0]], y=[Ax_hat_3d[1]], z=[Ax_hat_3d[2]],
        mode='markers+text',
        marker=dict(size=10, color='#1f77b4', symbol='circle'),
        text=['Ax̂'], textposition='bottom left',
        textfont=dict(size=18, color='#1f77b4', family='Arial Black'),
        showlegend=False, hoverinfo='text', hovertext='Projection Ax̂'
    ))
    
    # === Error vector (dashed line from Ax̂ to b) ===
    fig.add_trace(go.Scatter3d(
        x=[Ax_hat_3d[0], b_3d[0]], y=[Ax_hat_3d[1], b_3d[1]], z=[Ax_hat_3d[2], b_3d[2]],
        mode='lines',
        line=dict(color='#e74c3c', width=5, dash='dash'),
        name='b - Ax̂ (error vector)'
    ))
    
    # === Origin marker ===
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=['O'], textposition='bottom left',
        textfont=dict(size=14, color='black'),
        showlegend=False, hoverinfo='text', hovertext='Origin'
    ))
    
    # === Layout ===
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Component 1',
                range=[-max_range, max_range],
                showgrid=True, gridcolor='rgba(180,180,180,0.4)',
                zerolinecolor='rgba(0,0,0,0.4)', zerolinewidth=2,
                showbackground=True, backgroundcolor='rgba(245,245,250,0.95)'
            ),
            yaxis=dict(
                title='Component 2',
                range=[-max_range, max_range],
                showgrid=True, gridcolor='rgba(180,180,180,0.4)',
                zerolinecolor='rgba(0,0,0,0.4)', zerolinewidth=2,
                showbackground=True, backgroundcolor='rgba(245,250,245,0.95)'
            ),
            zaxis=dict(
                title='Component 3',
                range=[-max_range, max_range],
                showgrid=True, gridcolor='rgba(180,180,180,0.4)',
                zerolinecolor='rgba(0,0,0,0.4)', zerolinewidth=2,
                showbackground=True, backgroundcolor='rgba(250,245,245,0.95)'
            ),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(100,100,100,0.3)',
            borderwidth=1,
            font=dict(size=11)
        ),
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === Mathematical relationship ===
    st.markdown("---")
    
    # Build the linear combination LaTeX
    parts = []
    for i, coef in enumerate(x_hat):
        sign = '+' if coef >= 0 and i > 0 else ''
        parts.append(f"{sign}{coef:.2f} \\cdot v_{{{i+1}}}")
    
    st.markdown("**Linear Combination (Projection):**")
    st.latex(f"A\\hat{{x}} = {' '.join(parts)}")
    
    # Info boxes
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        **What you see:**
        - **Purple plane**: Column space Col(A)
        - **Colored arrows**: Column vectors v₁, v₂, ... ({num_cols} total)
        - **Red ◆**: Target vector b
        - **Blue ●**: Projection Ax̂ (on the plane)
        """)
    with col2:
        st.info("""
        **Key Insight:**
        The dashed error line (b - Ax̂) is **perpendicular** 
        to the column space plane. This is why least squares 
        gives the best approximation!
        
        🖱️ **Drag to rotate, scroll to zoom**
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
