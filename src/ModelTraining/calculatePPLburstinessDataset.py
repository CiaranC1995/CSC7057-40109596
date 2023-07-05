# Computing Perplexity & Burstiness scores for GPT-Intro_Wiki dataset

import pandas as pd
from src.ClassificationMetrics.perplexity import PerplexityBurstiness as pb
import time

start_time = time.time()

csv = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro.csv")

csv = csv[csv["generated_intro_len"] >= 150]

csv = csv[["wiki_intro", "generated_intro"]]

dataframe = pd.DataFrame()

dataframe["text_to_analyse"] = csv["wiki_intro"].to_list() + csv["generated_intro"].to_list()

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
# dataframe.drop("text_to_analyse", axis=1, inplace=True)

# Exporting the dataframe to a CSV file
dataframe.to_csv("NEW_perplexity_burstiness_scores.csv", index=True)

print("Time Elapsed: {:.2f}s".format(time.time() - start_time))
