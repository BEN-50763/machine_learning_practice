# %% [markdown]
# # Jupyter notebook 1.2: Applying calculus in Python
# 
# Calculus provides essential tools for understanding and optimising functions. The derivative measures the rate of change of a function with respect to a single variable, helping to identify maxima, minima and points of inflection. For functions of multiple variables, partial differentiation calculates the derivative with respect to one variable while keeping the others constant, enabling the analysis of complex systems.
# 
# The norm quantifies the size or length of a vector and is often used to measure distances or magnitudes in multidimensional spaces.
# 
# Based on these concepts, gradient descent is an iterative optimisation algorithm that uses the gradient (a vector of partial derivatives) to move towards a function’s minimum by stepping in the direction of steepest descent. It is widely used in ML and numerical optimisation.

# %% [markdown]
# ## 1. Derivatives 

# %%
from sympy import symbols, diff

x = symbols('x')
f = x**7  # Function
f_derivative = diff(f, x)
print("Derivative of x^7:", f_derivative)

# %% [markdown]
# ## 2. Partial derivatives

# %%
from sympy import symbols, sin, diff

# Define the variables
x, y, z = symbols('x y z')

# Define the function
f = 4*x*y + x*sin(z) + x**3 + z**8*y

# Calculate the partial derivatives
f_partial_x = diff(f, x)
f_partial_y = diff(f, y)
f_partial_z = diff(f, z)

# Print the partial derivatives
print("Partial derivative with respect to x:", f_partial_x)
print("Partial derivative with respect to y:", f_partial_y)
print("Partial derivative with respect to z:", f_partial_z)

# Evaluate the partial derivatives at specific points
# For example, evaluate at x=1, y=2, z=3
print("\nEvaluating partial derivatives at x=1, y=2, z=3:")
print("Partial derivative with respect to x:", f_partial_x.subs({x: 1, y: 2, z: 3}))
print("Partial derivative with respect to y:", f_partial_y.subs({x: 1, y: 2, z: 3}))
print("Partial derivative with respect to z:", f_partial_z.subs({x: 1, y: 2, z: 3}))


# %% [markdown]
# ## 3. Chain rule

# %%
g = (x**2 + 1)**3
g_derivative = diff(g, x)
print("Derivative using Chain Rule:", g_derivative)

# %% [markdown]
# ## 4. Norms
# 
# In mathematics, a norm is a way to measure the size or length of a vector. Think of it like measuring how far a point is from the origin (0, 0) in a graph. Norms are important because they help you understand how big or small a vector is, which is crucial in many mathematical and computational applications.
# 
# ### Common types of norms:
# 
# - **L1 norm (Manhattan distance)**: sums the values of all elements in a vector
# 
# - **L2 norm (Euclidean distance)**: takes the square root of the sum of the squares of all elements
# 
# - **L∞ norm (max norm)**: the maximum absolute value among all elements

# %% [markdown]
# The `NumPy` library uses the function `.linalg.norm()` to calculate the norm (magnitude or length) of a vector:
# 
# - The `ord=1` indicates that it is an L1-norm.
# 
# - The default is an L2-norm. 
# 
# - For L∞ norms, the `ord` is `ord=np.inf`.

# %%
import numpy as np

# Define a vector
vector = np.array([1, 2, 3])

# Calculate the L1 norm
l1_norm = np.linalg.norm(vector, ord=1)
print(f"L1 Norm: {l1_norm}")

# Calculate the L2 norm (default)
l2_norm = np.linalg.norm(vector)
print(f"L2 Norm: {l2_norm}")

# Calculate the L∞ norm
inf_norm = np.linalg.norm(vector, ord=np.inf)
print(f"L∞ Norm: {inf_norm}")


# %% [markdown]
# ## 5. Gradient descent
# 
# In this practice exercise, you will apply gradient descent to minimise the function `f(x)=(x−3)2`, which has its minimum at `x = 3`.
# 
# The gradient of the function is `f′(x) = 2(x − 3)`, and the value of `x` is updated by moving in the direction opposite to the gradient. In this case, the learning rate (`α`) is set to 0.1.
# 
# You can vary the number of iterations to observe how `x` converges towards the minimum.
# 
# The updated values of `x` after each iteration will be stored in a history list, allowing you to track how `x` changes over time.
# 
# Feel free to experiment with the number of iterations and the learning rate to explore their effects on the convergence rate.

# %%
# Given function: f(x) = (x - 3)^2
# Derivative: f'(x) = 2(x - 3)
# Initial value: x0 = 10
# Learning rate: alpha = 0.1
# Perform three iterations
# You can vary the number of iterations to analyse the convergence
# Initialisation
x = 10  # initial guess
alpha = 0.1  # learning rate
iterations = 3  # number of iterations
# Perform gradient descent updates
history = [x]
for _ in range(iterations):
    gradient = 2 * (x - 3)  # Compute the derivative f'(x)
    x = x - alpha * gradient  # Update x
    history.append(x)
history



