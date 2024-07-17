import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Define the experiments and categories
configurations = [
    "Short - Easy env",
    "Short - MNIST",
    "Short - CIFAR10",
    "Long - Easy env",
    "Long - MNIST",
    "Long - CIFAR10"
]

experimental_variables = [
    "RNN",
    "LSTM",
    "GRU",
    "LSTM+MSELoss",
    "LSTM+SmoothL1Loss"
]

# Generate random rewards for each experiment and each variable
data = []
for config in configurations:
    for var in experimental_variables:
        # average reward between 0 and 100
        reward = np.random.rand(10).mean() * 100
        data.append([config, var, reward])

# Create a DataFrame
df = pd.DataFrame(
    data, columns=["Configuration", "Experimental Variable", "Average Reward"])

# Plot 1: Short and Long Easy Environments with RNN, LSTM, and GRU
filtered_data_easy_env = df[(df["Configuration"].isin(["Short - Easy env", "Long - Easy env"])) &
                            (df["Experimental Variable"].isin(["RNN", "LSTM", "GRU"]))]

plt.figure(figsize=(10, 6))
sns.barplot(data=filtered_data_easy_env, x="Configuration",
            y="Average Reward", hue="Experimental Variable")
plt.title("Average Reward Achieved in Easy Environments")
plt.xticks(rotation=45)
plt.legend(title="Experimental Variable")
plt.tight_layout()
plt.show()

# Plot 2: All Conditions with RNN, LSTM, and GRU
filtered_data_all_conditions = df[(
    df["Experimental Variable"].isin(["RNN", "LSTM"]))]

plt.figure(figsize=(12, 8))
sns.barplot(data=filtered_data_all_conditions, x="Configuration",
            y="Average Reward", hue="Experimental Variable")
plt.title("Average Reward Achieved Across All Conditions")
plt.xticks(rotation=45)
plt.legend(title="Experimental Variable")
plt.tight_layout()
plt.show()
