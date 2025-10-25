# %% [markdown]
# ## Jupyter notebook 1.3: Applying probability and statistics in Python
# 
# This Jupyter notebook covers the normal and binomial distributions, focusing on their mean and variance.

# %%
#Normal distribution
import numpy as np
import matplotlib.pyplot as plt
# Simulate a normal distribution
data = np.random.normal(loc=5, scale=2, size=1000) # mean 5, std dev 2

# Calculate the expectation (mean) and variance
mean = np.mean(data)
variance = np.var(data)

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot a histogram of the data

plt.hist(data, bins=30)
plt.show()

# %%
# Binomial distribution
n, p = 10, 0.5  # number of trials, probability of success
binomial_data = np.random.binomial(n, p, size=1000)
binomial_mean = np.mean(binomial_data)
binomial_variance = np.var(binomial_data)

print(f"Binomial Distribution - Mean: {binomial_mean}, Variance: {binomial_variance}")

# Plot the histograms
fig, ax = plt.subplots(1, figsize=(12, 5))
ax.hist(binomial_data, bins=30, color='lightcoral', edgecolor='black')
ax.set_title("Binomial Distribution")
plt.show()




# %%
