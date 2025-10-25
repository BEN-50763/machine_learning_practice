# 3D vector operations and visualisation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define 3D vectors
m1 = np.array([[2, 3, 1], [4, 0, 5]])
m2 = np.array([[1, 4], [2, 3], [0,6]])

np.dot(m1, m2)

####

m3 = np.array([[2,1], [0,5]])

eigvals, eigvecs = np.linalg.eig(m3)