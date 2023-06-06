import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from TextPreprocessor.preprocessor import *

# Creating Instance of TextPreprocessor
preprocessor = TextPreprocessor()

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()


def calculate_perplexity(text, model, tokenizer):
    """Calculates the perplexity of a given input text."""
    inputs = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


def calculate_burstiness(perplexities):
    """Calculates the variation in perplexity (burstiness) of a given input text"""
    num_sentences = len(perplexities)
    avg_perplexity = sum(perplexities) / num_sentences
    burstiness = math.sqrt(sum((p - avg_perplexity) ** 2 for p in perplexities) / num_sentences)
    return burstiness


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenized_text = preprocessor.split_into_sentences(text_to_analyse)

# Remove new line characters from tokenized list of sentences
processed_tokenized_text = []
for sentence in tokenized_text:
    processed_sentence = preprocessor.remove_new_line_characters(sentence)
    processed_tokenized_text.append(processed_sentence)

# Calculate the individual PPL of each sentence
sentence_perplexities = []
for sentence in processed_tokenized_text:
    perplexity = calculate_perplexity(sentence, model, tokenizer)
    sentence_perplexities.append(perplexity)

# Calculate average PPL of whole text
print(f'Average text Perplexity : {sum(sentence_perplexities) / len(sentence_perplexities)}')
print(f'Burstiness of Text : {calculate_burstiness(sentence_perplexities)}')

# Print the sentence with the highest PPL
max_index = sentence_perplexities.index(max(sentence_perplexities))
print(
    f"Sentence with highest PPL of {round(max(sentence_perplexities), 2)} is sentence {max_index + 1} : "
    f"'{processed_tokenized_text[max_index]}'")
