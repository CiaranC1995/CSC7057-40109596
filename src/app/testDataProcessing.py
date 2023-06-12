"""Using the *** found on HuggingFace, which is a dataset containing 150k rows, to calculate
ClassificationMetrics with the final aim of training a machine learning binary classification model to distinguish
authorship of a given input text."""

import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.ClassificationMetrics import perplexity as pp

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Read in dataset from csv file
dataset = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Examples of Text\test.csv")

dataframe = pd.DataFrame()

dataframe["text_to_analyse"] = dataset["text"].to_list()

dataframe["perplexity"] = [pp.PerplexityBurstiness.calculate_perplexity(text, model, tokenizer) for text in dataframe["text_to_analyse"]]

dataframe["AI Generated"] = dataset["ai_generated"]

print(dataframe)
