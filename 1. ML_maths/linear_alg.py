# %% [markdown]
# # Jupyter notebook 1.1: Applying linear algebra in Python
# 
# ## 1. Vectors
# 
# Vectors are a fundamental concept in linear algebra, forming the foundation of vector spaces where operations such as addition and scalar multiplication are defined.

# %%
import numpy as np

def vector_addition(v1, v2):
    return np.add(v1, v2)

def vector_subtraction(v1, v2):
    return np.subtract(v1, v2)

def scalar_multiplication(v, scalar):
    return np.multiply(v, scalar)

def dot_product(v1, v2):
    return np.dot(v1, v2)

def cross_product(v1, v2):
    return np.cross(v1, v2)

# Example usage
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
scalar = 3

print("Addition:", vector_addition(v1, v2))
print("Subtraction:", vector_subtraction(v1, v2))
print("Scalar Multiplication:", scalar_multiplication(v1, scalar))
print("Dot Product:", dot_product(v1, v2))
print("Cross Product:", cross_product(v1, v2))


# %% [markdown]
# ## 2. Matrices
# 
# Matrices are two-dimensional arrays of numbers, arranged in rows and columns. They are a fundamental tool in linear algebra and play a crucial role in machine learning for representing and manipulating data. They are essential for data representation, linear transformations and efficient computations.
# 

# %%
# Creating and manipulating arrays using NumPy
scalar = np.array(2)
row_vector = np.array([1, 2, 3])  # Row vector
column_vector = np.array([[1], [2], [3]])  # Column vector
matrix = np.array([[1, 2], [3, 4]])  # 2x2 Matrix

# Display examples
print("Scalar:", scalar)
print("Row Vector:", row_vector)
print("Column Vector:\n", column_vector)
print("Matrix:\n", matrix)

# %% [markdown]
# ## Matrix operations

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C = A + B
print("Matrix Addition:\n", C)

# Scalar multiplication
scalar_mult = 2 * A
print("Scalar Multiplication:\n", scalar_mult)

#Matrix multiplication
dot_product = np.dot(A, B)
print("Matrix Multiplication:\n", dot_product)

# %% [markdown]
# ## Dot product

# %%
#Alternatively, the @ operator introduced in Python 3.5 is also used for matrix multiplication
dot_product = A @ B
print("Dot Product (Matrix Multiplication):\n", dot_product)


# %% [markdown]
# ## Computing eigenvalues and eigenvectors

# %%
import numpy as np

# Define the matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: \n", eigenvectors)

# Verify the relationship Av = λv for the first eigenvector
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0]

# Normalise the eigenvector for easier comparison
v1_normalized = v1 / np.linalg.norm(v1)

# Compute Av
Av = np.dot(A, v1_normalized)

# Compute λv
lambda_v = lambda1 * v1_normalized

print("\nVerification:")
print("Av: ", Av)
print("λv: ", lambda_v)


# %% [markdown]
# ## 4. Linear transformation
# 
# Some basic operations of linear transformations are:
# 
# - **Scaling**: multiplies each component of a vector by a scalar
# 
# - **Rotation**: rotates a vector by a certain angle around the origin
# 
# - **Reflection**: flips a vector across a certain axis
# 
# - **Projection**: projects a vector onto another vector or a subspace
# 

# %%
import numpy as np

# Original vector
v = np.array([1, 2, 3])

# Scaling factor
scale = 2

# Apply scaling transformation
scaled_v = v * scale

print("Original Vector:", v)
print("Scaled Vector:", scaled_v)


# %%
import numpy as np
import math

# Original vector
v = np.array([1, 0])

# Rotation angle in radians
theta = math.pi / 2  # 90 degrees

# Rotation matrix
rotation_matrix = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
])

# Apply rotation transformation
rotated_v = np.dot(rotation_matrix, v)

print("Original Vector:", v)
print("Rotated Vector:", rotated_v)


# %%
import numpy as np

# Original vector
v = np.array([1, 2])

# Reflection matrix (reflection across the x-axis)
reflection_matrix = np.array([
    [1, 0],
    [0, -1]
])

# Apply reflection transformation
reflected_v = np.dot(reflection_matrix, v)

print("Original Vector:", v)
print("Reflected Vector:", reflected_v)


# %%
import numpy as np

# Original vector
v = np.array([1, 2])

# Vector on to which to project
u = np.array([1, 0])

# Normalise u
u_normalized = u / np.linalg.norm(u)

# Projection formula
projected_v = np.dot(v, u_normalized) * u_normalized

print("Original Vector:", v)
print("Projected Vector:", projected_v)


# %% [markdown]
# ## Singular value decomposition (SVD)

# %%
import numpy as np

# Define a matrix A
A = np.array([[3, 1],
              [2, 2]])

# Perform singular value decomposition
U, sigma, VT = np.linalg.svd(A)

# Print the results
print("Original Matrix A:")
print(A)

print("\nLeft Singular Matrix U:")
print(U)

print("\nSingular Values (Sigma):")
print(sigma)

# Sigma needs to be converted to a diagonal matrix (for dimensionality matching)
Sigma_matrix = np.zeros_like(A, dtype=float)

# Fill the diagonal with the singular values
for i in range(len(sigma)):
    Sigma_matrix[i, i] = sigma[i]

print("\nSigma (Diagonal Matrix of Singular Values):")
print(Sigma_matrix)

print("\nRight Singular Matrix V^T:")
print(VT)

# Verify the decomposition by multiplying U, Sigma, and VT
A_reconstructed = np.dot(U, np.dot(Sigma_matrix, VT))

print("\nReconstructed Matrix A from U, Sigma, V^T:")
print(A_reconstructed)


# %%
import numpy as np

# Original matrix
A = np.array([[3, 1],
              [1, 3]], dtype='float64')

### Manual Computation ###
# Step 1: Compute A^T A
ATA = A.T @ A

# Step 2: Eigen decomposition of A^T A
eigenvalues, V = np.linalg.eig(ATA)

# Step 3: Sort eigenvalues/vectors (descending order)
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
V = V[:, sort_idx]

# Step 4: Compute the singular values and Σ matrix
sigma = np.sqrt(eigenvalues)
Sigma = np.diag(sigma)

# Step 5: Compute U matrix
U = np.zeros_like(A)
for i in range(V.shape[1]):
    U[:, i] = (A @ V[:, i]) / sigma[i]

### Verification ###
# Reconstruct A from the components
A_reconstructed = U @ Sigma @ V.T

# Compare it with NumPy's built-in SVD
U_np, S_np, VT_np = np.linalg.svd(A)

print("Manual Computation Results:")
print("U:\n", U.round(6))
print("Σ:\n", Sigma.round(6))
print("V.T:\n", V.T.round(6))
print("\nReconstructed Matrix:\n", A_reconstructed.round(6))
#print("\nNumPy SVD Results:")
#print("U:\n", U_np.round(6))
#print("Σ:\n", np.diag(S_np).round(6))
#print("V.T:\n", VT_np.round(6))


# %% [markdown]
# ## LU decomposition

# %%
import numpy as np

def lu_decomposition(matrix):
    # Check whether the matrix is invertible
    if np.linalg.det(matrix) == 0:
        raise ValueError("Matrix is not invertible.")

    # Extract elements of the matrix
    a, b = matrix[0, :]
    c, d = matrix[1, :]

    # Calculate elements of L and U
    l11, l21, l22 = 1, c / a, 1
    u11, u12, u22 = a, b, d - (c * b) / a

    # Construct the L and U matrices
    L = np.array([[l11, 0], [l21, l22]])
    U = np.array([[u11, u12], [0, u22]])

    return L, U

# Example usage
matrix = np.array([[4, 12], [6, 26]])
try:
    L, U = lu_decomposition(matrix)
    print("Matrix A:")
    print(matrix)
    print("\nLower Triangular Matrix L:")
    print(L)
    print("\nUpper Triangular Matrix U:")
    print(U)
except ValueError as e:
    print(e)



