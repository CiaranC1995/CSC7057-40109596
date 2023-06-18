import datetime
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv(r"perplexity_burstiness_scores.csv")

dataframe = pd.DataFrame()

dataframe["perplexity"] = csv['perplexity']
dataframe["burstiness"] = csv['burstiness']
dataframe["AI Generated"] = csv['AI Generated']

# Plotting perplexity
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(dataframe.index, dataframe["perplexity"], label="Perplexity")
plt.xlabel("Index")
plt.ylabel("Perplexity")
plt.title("Perplexity Plot")

# Plotting burstiness
plt.subplot(2, 1, 2)
plt.plot(dataframe.index, dataframe["burstiness"], label="Burstiness")
plt.xlabel("Index")
plt.ylabel("Burstiness")
plt.title("Burstiness Plot")

# Adding AI Generated column as color-coded markers
ai_generated_markers = ['r' if val == 1 else 'b' for val in dataframe["AI Generated"]]
plt.scatter(dataframe.index, dataframe["burstiness"], c=ai_generated_markers, label="AI Generated")

# Adding legend
plt.legend()

# Adjusting layout
plt.tight_layout()

# Displaying the plot
plt.show()

print("Program finished executing at:", datetime.datetime.now())
