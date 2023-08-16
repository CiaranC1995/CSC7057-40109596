import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, gamma, norm, weibull_min
from scipy.optimize import minimize

# Load the dataset from the CSV file
# data = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Training_X_test_data_analysis"
#                    r"\Correctly_Classified_Training_Test_Records\correctly_classified_records.csv")

data = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Documents\Data Visualisation\Distribtion Analysis\Correctly "
                   r"Classified\Correctly Classified as Human\Burstiness\outliersRemoved.csv")

# Function to calculate the SSE (Sum of Squared Errors) for the distribution fit
def sse(params, dist, data):
    _, dist_params = params[0], params[1:]
    dist_instance = dist(*dist_params)
    return np.sum((data - dist_instance.rvs(len(data))) ** 2)

# Function to calculate Cullen's parameters (a and b)
def cullen_params(data):
    mean = data.mean(numeric_only=True)  # Explicitly set numeric_only=True
    std_dev = data.std(numeric_only=True)  # Explicitly set numeric_only=True
    skewness = data.skew(numeric_only=True)  # Explicitly set numeric_only=True
    a = 2 * skewness / std_dev
    b = mean - 2 * std_dev * skewness
    return a, b

# Calculate Cullen's parameters
cullen_a, cullen_b = cullen_params(data['burstiness'])

# Plot Cullen and Frey graph
plt.scatter(cullen_a, cullen_b)
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.xlabel("Cullen a")
plt.ylabel("Cullen b")
plt.title("Cullen and Frey Graph")
plt.grid(True)
plt.show()

# Define candidate distributions for fitting the data along with their names
distributions = {
    "gamma": gamma,
    "norm": norm,
    "skewnorm": skewnorm,
    "weibull_min": weibull_min,
}

# Find the best distribution fit for the data
best_fit = None
best_params = None
best_sse = float('inf')

for dist_name, dist in distributions.items():
    params = dist.fit(data['burstiness'])
    sse_value = sse([0] + list(params), dist, data['burstiness'])

    if sse_value < best_sse:
        best_fit = dist
        best_params = params
        best_sse = sse_value

print("Best distribution fit:", best_fit.name if best_fit != gamma else "gamma")
print("Best distribution parameters:", best_params)

burstiness = data['burstiness']

# Define the desired confidence level
confidence_level = 95  # 90% of values fall above this probability

# Calculate the lower threshold using the desired confidence level
lower_threshold = np.percentile(burstiness, confidence_level)

print(f"Lower threshold for correctly classified inputs at {confidence_level}th percentile: {lower_threshold}")

# Plot histogram of the data
plt.hist(data['burstiness'], bins=20, density=True, alpha=0.6, label="Data Histogram")

# Generate x values for the PDF
x = np.linspace(data['burstiness'].min(), data['burstiness'].max(), 100)

# Plot the PDF of the best-fit distribution
best_dist_instance = best_fit(*best_params)
plt.plot(x, best_dist_instance.pdf(x), 'r', label="Best-fit Distribution")

# Plot the lower threshold value as a vertical line
plt.axvline(x=lower_threshold, color='g', linestyle='--', label=f"{confidence_level}th Percentile Threshold")

plt.xlabel("Data Values")
plt.ylabel("Probability Density")
plt.title("Data Histogram with Best-fit Distribution")
plt.legend()
plt.grid(True)
plt.show()