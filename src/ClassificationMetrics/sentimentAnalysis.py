from nltk.sentiment import SentimentIntensityAnalyzer
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_SpaceStory.txt') as f:
    text_to_analyse = f.read()


class SentimentAnalysis:

    def __init__(self):
        pass
    
    @staticmethod
    def perform_sentiment_analysis(input_text):
        """Calculates sentiment score of an input text"""
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(input_text)
        compound_score = sentiment_scores['compound']
        return compound_score

    @staticmethod
    def perform_average_sentiment_analysis(input_text):
        """Calculates average sentiment score of an input text"""
        tokenized_text = pp.TextPreprocessor.split_into_sentences(input_text)
        # Remove new line characters from tokenized list of sentences
        processed_tokenized_text = [pp.TextPreprocessor.remove_new_line_chars(sentence) for sentence in tokenized_text]
        sentence_sentiment_scores = [SentimentAnalysis.perform_sentiment_analysis(sentence) for sentence in processed_tokenized_text]
        avg_sentiment_score = sum(sentence_sentiment_scores) / len(sentence_sentiment_scores)
        # Unsure if average sentiment score is the right way to go
        return avg_sentiment_score


# print(f'Sentiment Score : {SentimentAnalysis.perform_sentiment_analysis(text_to_analyse)}')
# print(f'Average Sentiment Score : {SentimentAnalysis.perform_average_sentiment_analysis(text_to_analyse)}')
