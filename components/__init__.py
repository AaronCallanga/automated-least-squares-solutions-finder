# Components package
from .inputs import render_data_points_input, render_matrix_input, render_linear_system_input, parse_data_points, parse_matrix_input
from .display import (
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
