# Automated Least Squares Solutions Finder

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automated-least-squares-solutions-finder.streamlit.app/)

An interactive, beginner-friendly calculator that demonstrates the **Least Squares Approximation** step-by-step using the Normal Equation.

$$A^T A \hat{x} = A^T \vec{b}$$

🔗 **Live Demo:** [https://automated-least-squares-solutions-finder.streamlit.app/](https://automated-least-squares-solutions-finder.streamlit.app/)

---

## Features

- **Three Input Modes:**
  - **Data Points** – Enter x,y coordinates for line fitting
  - **Linear System** – Enter equations like `a + 2b = 4`
  - **Matrix Input** – Enter matrix A and vector b directly (supports fractions!)

- **Step-by-Step Solution** – See each calculation step with LaTeX notation

- **Interactive 3D Visualization** – Explore the geometric interpretation of least squares with Plotly

- **Fraction/Decimal Toggle** – View results as fractions or decimals

- **Educational Content** – Learn what least squares is and why it works

---

## How It Works

1. Select an input mode (Data Points, Linear System, or Matrix)
2. Choose display format (Decimal or Fraction)
3. Enter your data
4. Click **Calculate**
5. View the step-by-step solution
6. Explore the visualization and error analysis

---

## Project Structure

```
linalg/
├── app.py                    # Main application entry point
├── requirements.txt          # Python dependencies
├── styles/
│   └── css.py               # Custom CSS styles
├── utils/
│   ├── least_squares.py     # Core calculation logic
│   └── formatting.py        # Number formatting (decimal/fraction)
├── components/
│   ├── inputs.py            # Input form components
│   └── display.py           # Output display components
└── references/              # Reference images
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AaronCallanga/automated-least-squares-solutions-finder.git
cd automated-least-squares-solutions-finder

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Technologies Used

- **Streamlit** – Web application framework
- **NumPy** – Numerical computations
- **Matplotlib** – 2D visualizations
- **Plotly** – Interactive 3D visualizations

---

## Group 1 | BSCS 2-3 | Automated Least Squares Solutions Finder 

Names:
- Mark Jason Manlapaz
- Jomar Escala
- Christine Rio
- Anthony Bonito
- Aaron Dave Callanga

---

## License

This project is for educational purposes.
