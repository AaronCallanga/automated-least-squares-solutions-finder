import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Linear Algebra App",
    page_icon="📐",
    layout="wide"
)

# Main title
st.title("📐 Linear Algebra Application")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
st.sidebar.info("Welcome to the Linear Algebra App!")

# Main content area
st.header("Welcome!")
st.write("Start building your linear algebra application here.")

# Example: Simple plot
st.subheader("Example Plot")
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.set_title("Sine Wave")
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, NumPy, and Matplotlib")
