# Computing Perplexity & Burstiness scores for GPT-Intro_Wiki dataset

import pandas as pd
from src.ClassificationMetrics.perplexity import PerplexityBurstiness as pb
import matplotlib.pyplot as plt

csv = pd.read_csv(r"C:\Users\ccase\Desktop\GPT-wiki-intro.csv")
# csv = pd.read_csv("test.csv")

csv = csv[csv["generated_intro_len"] >= 150]

csv = csv[["wiki_intro", "generated_intro"]]

dataframe = pd.DataFrame()

dataframe["text_to_analyse"] = csv["wiki_intro"].to_list() + csv["generated_intro"].to_list()

# dataframe["text_to_analyse"] = csv["Text"].to_list()

results = [pb.process_text_ppl_burstiness(text) for text in dataframe["text_to_analyse"]]
perplexity_values = []
burstiness_values = []

for result in results:
    perplexity_values.append(result["avg_text_ppl"])
    burstiness_values.append(result["text_burstiness"])

dataframe["perplexity"] = perplexity_values
dataframe["burstiness"] = burstiness_values

dataframe["AI Generated"] = [0 for i in range(len(csv))] + [1 for i in range(len(csv))]

# Drop the "text_to_analyse" column
dataframe.drop("text_to_analyse", axis=1, inplace=True)

# Exporting the dataframe to a CSV file
dataframe.to_csv("perplexity_burstiness_scores.csv", index=True)

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
