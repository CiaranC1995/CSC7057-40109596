import string
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()


def split_text_into_sentences(input_text):
    sentences = sent_tokenize(input_text, language='English')
    # Ensures that each tokenized sentence contains at least one valid word
    filtered_sentences = [sentence for sentence in sentences if re.search(r'\w', sentence)]
    return filtered_sentences


def calculate_num_of_characters(input_text):
    """Calculates the number of characters in an input text"""
    return len(input_text)


def remove_new_line_characters(input_text):
    processed_text = input_text.replace('\n', '')
    return processed_text
