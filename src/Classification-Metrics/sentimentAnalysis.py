from nltk.sentiment import SentimentIntensityAnalyzer
from preprocessor import *

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_SpaceStory.txt') as f:
    text_to_analyse = f.read()


def perform_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score


tokenized_text = split_text_into_sentences(text_to_analyse)

# Remove new line characters from tokenized list of sentences
processed_tokenized_text = []
for sentence in tokenized_text:
    processed_sentence = remove_new_line_characters(sentence)
    processed_tokenized_text.append(processed_sentence)

sentence_sentiment_scores = []
for sentence in processed_tokenized_text:
    sentiment_score = perform_sentiment_analysis(sentence)
    # print(sentiment_score)
    sentence_sentiment_scores.append(sentiment_score)

# Unsure if average sentiment score is the right way to go
print(f'Average Sentiment Score : {sum(sentence_sentiment_scores) / len(sentence_sentiment_scores)}')


