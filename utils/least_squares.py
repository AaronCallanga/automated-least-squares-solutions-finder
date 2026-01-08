"""
Least Squares Solver Module

This module contains the core mathematical logic for computing
least squares solutions using the Normal Equation:

    x̂ = (AᵀA)⁻¹Aᵀb

The solver provides step-by-step calculations for educational purposes.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LeastSquaresResult:
    """
    Stores all intermediate and final results of the least squares calculation.
    
    Attributes:
        A: Original matrix A (m x n)
        b: Original vector b (m x 1)
        A_transpose: Transpose of A (n x m)
        ATA: Product of Aᵀ and A (n x n)
        ATA_inverse: Inverse of AᵀA (n x n)
        ATb: Product of Aᵀ and b (n x 1)
        x_hat: The least squares solution (n x 1)
        predicted: Predicted values Ax̂ (m x 1)
        residuals: Errors b - Ax̂ (m x 1)
        sse: Sum of Squared Errors
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        is_valid: Whether the calculation was successful
        error_message: Error message if calculation failed
    """
    A: np.ndarray
    b: np.ndarray
    A_transpose: Optional[np.ndarray] = None
    ATA: Optional[np.ndarray] = None
    ATA_inverse: Optional[np.ndarray] = None
    ATb: Optional[np.ndarray] = None
    x_hat: Optional[np.ndarray] = None
    predicted: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    sse: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    is_valid: bool = True
    error_message: str = ""


class LeastSquaresSolver:
    """
    Solves the least squares problem using the Normal Equation method.
    
    The Normal Equation: x̂ = (AᵀA)⁻¹Aᵀb
    
    This solver breaks down the calculation into steps:
        Step 1: Compute Aᵀ (transpose)
        Step 2: Compute AᵀA (matrix multiplication)
        Step 3: Compute (AᵀA)⁻¹ (matrix inverse)
        Step 4: Compute Aᵀb (matrix-vector multiplication)
        Step 5: Compute x̂ = (AᵀA)⁻¹Aᵀb (final solution)
        Step 6: Compute residuals and error metrics
    
    Example:
        >>> solver = LeastSquaresSolver()
        >>> A = np.array([[1, 1], [2, 1], [3, 1]])
        >>> b = np.array([2, 3, 5])
        >>> result = solver.solve(A, b)
        >>> print(result.x_hat)  # [1.5, 0.333...]
    """
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> LeastSquaresResult:
        """
        Solve the least squares problem Ax ≈ b.
        
        Args:
            A: Coefficient matrix of shape (m, n) where m >= n
            b: Target vector of shape (m,)
        
        Returns:
            LeastSquaresResult containing all intermediate steps and final solution
        """
        result = LeastSquaresResult(A=A, b=b)
        
        try:
            # Step 1: Compute transpose
            result.A_transpose = self._compute_transpose(A)
            
            # Step 2: Compute AᵀA
            result.ATA = self._compute_ATA(result.A_transpose, A)
            
            # Step 3: Compute (AᵀA)⁻¹
            result.ATA_inverse = self._compute_inverse(result.ATA)
            
            # Step 4: Compute Aᵀb
            result.ATb = self._compute_ATb(result.A_transpose, b)
            
            # Step 5: Compute x̂ = (AᵀA)⁻¹Aᵀb
            result.x_hat = self._compute_solution(result.ATA_inverse, result.ATb)
            
            # Step 6: Compute residuals and error metrics
            result.predicted = self._compute_predicted(A, result.x_hat)
            result.residuals = self._compute_residuals(b, result.predicted)
            result.sse, result.mse, result.rmse = self._compute_error_metrics(result.residuals)
            
        except np.linalg.LinAlgError as e:
            result.is_valid = False
            result.error_message = f"Matrix is singular (not invertible): {str(e)}"
        except Exception as e:
            result.is_valid = False
            result.error_message = f"Calculation error: {str(e)}"
        
        return result
    
    def _compute_transpose(self, A: np.ndarray) -> np.ndarray:
        """Step 1: Compute Aᵀ (transpose of A)."""
        return A.T
    
    def _compute_ATA(self, A_transpose: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Step 2: Compute AᵀA (matrix multiplication)."""
        return A_transpose @ A
    
    def _compute_inverse(self, ATA: np.ndarray) -> np.ndarray:
        """Step 3: Compute (AᵀA)⁻¹ (matrix inverse)."""
        return np.linalg.inv(ATA)
    
    def _compute_ATb(self, A_transpose: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Step 4: Compute Aᵀb (matrix-vector multiplication)."""
        return A_transpose @ b
    
    def _compute_solution(self, ATA_inverse: np.ndarray, ATb: np.ndarray) -> np.ndarray:
        """Step 5: Compute x̂ = (AᵀA)⁻¹Aᵀb (final solution)."""
        return ATA_inverse @ ATb
    
    def _compute_predicted(self, A: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """Compute predicted values: Ax̂."""
        return A @ x_hat
    
    def _compute_residuals(self, b: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Compute residuals: b - Ax̂."""
        return b - predicted
    
    def _compute_error_metrics(self, residuals: np.ndarray) -> tuple[float, float, float]:
        """
        Compute error metrics from residuals.
        
        Returns:
            Tuple of (SSE, MSE, RMSE)
        """
        sse = float(np.sum(residuals ** 2))
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        return sse, mse, rmse


def create_design_matrix(x_values: np.ndarray) -> np.ndarray:
    """
    Create the design matrix A for linear regression (y = mx + c).
    
    For fitting a line y = mx + c, we need:
        A = [[x₁, 1],
             [x₂, 1],
             [x₃, 1],
             ...]
    
    Args:
        x_values: Array of x coordinates
    
    Returns:
        Design matrix A of shape (n, 2)
    """
    return np.column_stack([x_values, np.ones(len(x_values))])
