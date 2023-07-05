import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp

tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()


class PerplexityBurstiness:

    def __init__(self):
        pass

    @staticmethod
    def calculate_perplexity(input_text, model, tokenizer):
        """Calculates the perplexity of a given input text."""
        inputs = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
        ppl = torch.exp(loss)
        return ppl.item()

    @staticmethod
    def calculate_burstiness(perplexities):
        """Calculates the variation in perplexity (burstiness) of a given input text"""
        num_sentences = len(perplexities)
        avg_perplexity = sum(perplexities) / num_sentences
        burstiness = math.sqrt(sum((p - avg_perplexity) ** 2 for p in perplexities) / num_sentences)
        return burstiness

    @staticmethod
    def process_text_ppl_burstiness(input_text):
        """Processes the average perplexity and burstiness of an input text"""
        tokenized_text = pp.TextPreprocessor.split_into_sentences(input_text)

        # Calculate the individual PPL of each sentence
        sentence_perplexities = []

        for sentence in tokenized_text:
            perplexity = PerplexityBurstiness.calculate_perplexity(sentence, model_gpt, tokenizer_gpt)
            sentence_perplexities.append(perplexity)

        # Calculate average PPL of whole text
        avg_text_ppl = sum(sentence_perplexities) / len(sentence_perplexities)
        text_burstiness = PerplexityBurstiness.calculate_burstiness(sentence_perplexities)

        return {
            'avg_text_ppl': avg_text_ppl,
            'text_burstiness': text_burstiness
        }

