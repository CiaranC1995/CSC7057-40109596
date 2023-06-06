from nltk.tokenize import sent_tokenize
import re

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()


class TextPreprocessor:

    def __init__(self):
        pass

    @staticmethod
    def split_into_sentences(input_text):
        """Separates a piece of text into sentences."""
        sentences = sent_tokenize(input_text, language='English')
        # Ensures that each tokenized sentence contains at least one valid word
        processed_sentences = [sentence for sentence in sentences if re.search(r'\w', sentence)]
        return processed_sentences

    @staticmethod
    def num_of_chars(input_text):
        """Calculates the number of characters in an input text"""
        return len(input_text)

    @staticmethod
    def remove_new_line_characters(input_text):
        """Removes new line characters from a piece of text."""
        processed_text = input_text.replace('\n', '')
        return processed_text
