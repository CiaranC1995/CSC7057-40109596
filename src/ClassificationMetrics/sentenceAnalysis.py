import re
import nltk
import matplotlib.pyplot as plt
import syllapy
from TextPreprocessor import preprocessor as pp

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    human_text_to_analyse = f.read()

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_SpaceStory.txt') as f:
    ai_text_to_analyse = f.read()


class SentenceAnalysis:

    def __init__(self):
        pass

    @staticmethod
    def num_total_words(input_text):
        """Calculates the total number of words in an input text"""
        processed_input_text = pp.TextPreprocessor.remove_punctuation(input_text)
        words = nltk.word_tokenize(processed_input_text)
        return len(words)

    @staticmethod
    def sentence_lengths(input_text):
        """Returns an array containing the number of words in each sentence of an input text"""
        sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        return [SentenceAnalysis.num_total_words(s) for s in sentences]

    @staticmethod
    def average_sentence_length_words(input_text):
        """Calculates the average number of words per sentence in an input text"""
        sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        return SentenceAnalysis.num_total_words(input_text) / len(sentences)

    @staticmethod
    def average_sentence_length_chars(input_text):
        """Calculates the average number of characters per sentence in an input text"""
        sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        total_chars = sum(len(sentence) for sentence in sentences)
        return total_chars / len(sentences)

    @staticmethod
    def display_individual_sentence_stats(input_text):
        """Displays stats on each individual sentence within an input text"""
        sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        print('----')
        for s in sentences:
            print(s)
            print(f'Number of words in sentence : {SentenceAnalysis.num_total_words(s)}')
            print(f'Number of chars in sentence : {len(s)}')
            print('----')

    @staticmethod
    def display_stats_for_whole_text(input_text):
        """Displays stats for an input text"""

        min_words = min(SentenceAnalysis.sentence_lengths(input_text))
        max_words = max(SentenceAnalysis.sentence_lengths(input_text))

        print(f'Number of Words in text : {SentenceAnalysis.num_total_words(input_text)}')
        print(f'Average number of words per sentence : {round(SentenceAnalysis.average_sentence_length_words(input_text), 3)}')
        print(f'Average number of chars per sentence : {round(SentenceAnalysis.average_sentence_length_chars(input_text), 3)}')
        print(f'Minimum number of words per sentence : {min_words}')
        print(f'Maximum number of words per sentence : {max_words}')
        print(f'Range of sentence lengths in input text : {max_words - min_words}')

    @staticmethod
    def plot_sentence_length_distribution(input_text):
        """Plots the sentence length distribution in a histogram"""
        lengths = SentenceAnalysis.sentence_lengths(input_text)

        plt.hist(lengths, bins=20)
        plt.xlabel('Sentence Length')
        plt.ylabel('Frequency')
        plt.title('Sentence Length Distribution')
        plt.show()

    @staticmethod
    def calculate_flesch_kincaid(input_text):
        """Calculates a Flesch-Kincaid readability score for an input text"""
        sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        num_sentences = len(sentences)
        num_words = SentenceAnalysis.num_total_words(input_text)
        processed_input_text = pp.TextPreprocessor.remove_punctuation(input_text)
        words = nltk.word_tokenize(processed_input_text)
        # Syllapy 3rd party library used to calculate syllables in the words
        num_syllables = sum(syllapy.count(word) for word in words)
        # Calculate the Flesch-Kincaid Score
        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        return score

    @staticmethod
    def calculate_coleman_liau(input_text):
        """Calculates a Coleman-Liau readability index for an input text. This index represents an approximation of the
        U.S. grade level thought necessary to comprehend the text."""
        sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        num_sentences = len(sentences)
        num_words = SentenceAnalysis.num_total_words(input_text)
        num_letters = len(re.sub(r'[^a-zA-Z0-9]', '', input_text))
        # Calculate the Coleman-Liau Index formula
        letters_per_100_words = (num_letters / num_words) * 100
        sentences_per_100_words = (num_sentences / num_words) * 100
        index = 0.0588 * letters_per_100_words - 0.296 * sentences_per_100_words - 15.8
        return index
