import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Least Squares Calculator",
    page_icon="📐",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .step-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .formula-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        margin: 1rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📐 Least Squares Calculator</h1>', unsafe_allow_html=True)

# Introduction
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

st.markdown("---")

# Sidebar for input method selection
st.sidebar.header("⚙️ Settings")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["📊 Data Points (Easy)", "🔢 Matrix Input (Advanced)"]
)

# Main content based on input method
if input_method == "📊 Data Points (Easy)":
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
    
    # Parse inputs
    try:
        x_values = np.array([float(x.strip()) for x in x_input.replace(',', '\n').split('\n') if x.strip()])
        y_values = np.array([float(y.strip()) for y in y_input.replace(',', '\n').split('\n') if y.strip()])
        
        if len(x_values) != len(y_values):
            st.error("❌ X and Y must have the same number of values!")
            st.stop()
        
        if len(x_values) < 2:
            st.error("❌ Please enter at least 2 data points!")
            st.stop()
        
        # Create matrix A (for line fitting: y = mx + c)
        A = np.column_stack([x_values, np.ones(len(x_values))])
        b = y_values
        
        valid_input = True
    except ValueError as e:
        st.error(f"❌ Invalid input! Please enter numbers only. Error: {e}")
        valid_input = False

else:  # Matrix input
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
    
    # Parse inputs
    try:
        A = np.array([[float(val) for val in row.split()] for row in a_input.strip().split('\n') if row.strip()])
        b = np.array([float(val.strip()) for val in b_input.strip().split('\n') if val.strip()])
        
        if A.shape[0] != len(b):
            st.error(f"❌ Matrix A has {A.shape[0]} rows but vector b has {len(b)} elements. They must match!")
            st.stop()
        
        valid_input = True
    except ValueError as e:
        st.error(f"❌ Invalid input! Please check your matrix format. Error: {e}")
        valid_input = False

# Calculate button
st.markdown("---")
calculate = st.button("🚀 Calculate Least Squares Solution", type="primary", use_container_width=True)

if calculate and valid_input:
    st.markdown("---")
    st.header("📝 Step-by-Step Solution")
    
    # Step 1: Show the problem
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("Step 1️⃣: Understanding the Problem")
        st.write("We want to solve **Ax = b** using the least squares formula:")
        st.latex(r"\hat{x} = (A^T A)^{-1} A^T b")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Matrix A:**")
            st.write(A)
            st.write(f"Shape: {A.shape[0]} rows × {A.shape[1]} columns")
        with col2:
            st.write("**Vector b:**")
            st.write(b)
            st.write(f"Shape: {len(b)} elements")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Calculate A^T
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("Step 2️⃣: Calculate Aᵀ (Transpose of A)")
        st.write("The transpose flips rows and columns:")
        st.latex(r"A^T = \text{swap rows and columns of } A")
        A_T = A.T
        st.write("**Aᵀ =**")
        st.write(A_T)
        st.write(f"Shape changed from {A.shape} → {A_T.shape}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Calculate A^T * A
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("Step 3️⃣: Calculate AᵀA (Matrix Multiplication)")
        st.write("Multiply Aᵀ by A:")
        st.latex(r"A^T A = A^T \times A")
        ATA = A_T @ A
        st.write("**AᵀA =**")
        st.write(ATA)
        st.write(f"Result shape: {ATA.shape[0]} × {ATA.shape[1]} (square matrix)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Calculate (A^T * A)^-1
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("Step 4️⃣: Calculate (AᵀA)⁻¹ (Inverse)")
        st.write("Find the inverse of AᵀA:")
        st.latex(r"(A^T A)^{-1}")
        
        try:
            ATA_inv = np.linalg.inv(ATA)
            st.write("**(AᵀA)⁻¹ =**")
            st.write(ATA_inv)
            st.success("✅ Matrix is invertible!")
        except np.linalg.LinAlgError:
            st.error("❌ Matrix AᵀA is singular (not invertible). Cannot compute least squares.")
            st.stop()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 5: Calculate A^T * b
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("Step 5️⃣: Calculate Aᵀb")
        st.write("Multiply Aᵀ by b:")
        st.latex(r"A^T b = A^T \times b")
        ATb = A_T @ b
        st.write("**Aᵀb =**")
        st.write(ATb)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 6: Final calculation
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.subheader("Step 6️⃣: Calculate x̂ = (AᵀA)⁻¹Aᵀb")
        st.write("Multiply (AᵀA)⁻¹ by Aᵀb to get our solution:")
        st.latex(r"\hat{x} = (A^T A)^{-1} A^T b")
        x_hat = ATA_inv @ ATb
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Result
    st.markdown("---")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.header("🎯 Final Solution")
    st.write("**x̂ =**")
    
    for i, val in enumerate(x_hat):
        st.write(f"x_{i+1} = {val:.6f}")
    
    if input_method == "📊 Data Points (Easy)":
        st.markdown("---")
        st.subheader("📈 Best-Fit Line Equation:")
        m, c = x_hat[0], x_hat[1]
        st.latex(f"y = {m:.4f}x + {c:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization
    st.markdown("---")
    st.header("📊 Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and best-fit line
    ax1 = axes[0]
    if input_method == "📊 Data Points (Easy)":
        ax1.scatter(x_values, y_values, color='#667eea', s=100, zorder=5, label='Data Points', edgecolors='white', linewidth=2)
        x_line = np.linspace(min(x_values) - 0.5, max(x_values) + 0.5, 100)
        y_line = x_hat[0] * x_line + x_hat[1]
        ax1.plot(x_line, y_line, color='#38ef7d', linewidth=3, label=f'Best Fit: y = {x_hat[0]:.3f}x + {x_hat[1]:.3f}')
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('Data Points & Best-Fit Line', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.bar(range(len(x_hat)), x_hat, color='#667eea', edgecolor='white', linewidth=2)
        ax1.set_xlabel('Variable Index', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Solution Values (x̂)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(x_hat)))
        ax1.set_xticklabels([f'x_{i+1}' for i in range(len(x_hat))])
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Residuals
    ax2 = axes[1]
    predicted = A @ x_hat
    residuals = b - predicted
    
    if input_method == "📊 Data Points (Easy)":
        ax2.stem(x_values, residuals, linefmt='#764ba2', markerfmt='o', basefmt='gray')
        ax2.axhline(y=0, color='#38ef7d', linestyle='--', linewidth=2)
        ax2.set_xlabel('X', fontsize=12)
    else:
        ax2.stem(range(len(residuals)), residuals, linefmt='#764ba2', markerfmt='o', basefmt='gray')
        ax2.axhline(y=0, color='#38ef7d', linestyle='--', linewidth=2)
        ax2.set_xlabel('Observation Index', fontsize=12)
    
    ax2.set_ylabel('Residual (Error)', fontsize=12)
    ax2.set_title('Residuals (b - Ax̂)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Error analysis
    st.subheader("📉 Error Analysis")
    col1, col2, col3 = st.columns(3)
    
    sse = np.sum(residuals**2)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    with col1:
        st.metric("Sum of Squared Errors (SSE)", f"{sse:.6f}")
    with col2:
        st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
    with col3:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.6f}")
    
    st.info("💡 **Lower error values indicate a better fit!** The least squares method minimizes the Sum of Squared Errors (SSE).")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>📐 Least Squares Calculator | Built with Streamlit, NumPy & Matplotlib</p>
    <p>Perfect for learning linear algebra! 🎓</p>
</div>
""", unsafe_allow_html=True)
