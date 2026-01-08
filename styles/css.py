"""
Custom CSS styles for the Least Squares Calculator.
All styling is centralized here for easy maintenance.
"""
import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
        /* Main header gradient */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        
        /* Step boxes for calculations */
        .step-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            border-left: 4px solid #667eea;
            padding: 1rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
        }
        
        /* Formula display box */
        .formula-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.3rem;
            margin: 1rem 0;
        }
        
        /* Result display box */
        .result-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        
        /* Info card styling */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
