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
        clean_text = input_text.replace('\n', ' ')
        return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', clean_text)

    @staticmethod
    def num_of_chars(input_text):
        """Calculates the number of characters in an input text"""
        return len(input_text)

    @staticmethod
    def remove_new_line_chars(input_text):
        """Removes new line characters from a piece of text."""
        processed_text = input_text.replace('\n', '')
        return processed_text

    @staticmethod
    def remove_punctuation(input_text):
        """Removes all punctuation from a piece of text"""
        pattern = r'[^\w\s]'
        processed_text = re.sub(pattern, '', input_text)
        return processed_text

