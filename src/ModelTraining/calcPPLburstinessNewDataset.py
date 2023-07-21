import pandas as pd
from src.ClassificationMetrics.perplexity import PerplexityBurstiness as pb
import time

start_time = time.time()

csv = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\chatgpt-generated-text-corpus"
                  r"\new_dataset_github.csv")

results = [pb.process_text_ppl_burstiness(text) for text in csv["text_to_analyse"]]
perplexity_values = []
burstiness_values = []

for result in results:
    perplexity_values.append(result["avg_text_ppl"])
    burstiness_values.append(result["text_burstiness"])

csv["perplexity"] = perplexity_values
csv["burstiness"] = burstiness_values

csv["AI_Generated"] = [0 for i in range(126)] + [1 for i in range(126)]

csv.to_csv("NEW_DATASET_perplexity_burstiness_scores.csv", index=False)

print("Time Elapsed: {:.2f}s".format(time.time() - start_time))



