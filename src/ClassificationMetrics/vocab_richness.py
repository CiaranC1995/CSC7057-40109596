import nltk
from nltk.corpus import stopwords
from TextPreprocessor import preprocessor as pp


# Reading in human written text to analyse from external file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    human_text_to_analyse = f.read()

# Reading in AI written text to analyse from external file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_AgileSoftwareDevelopment.txt') as f:
    ai_text_to_analyse = f.read()


def calculate_total_vocab_richness(input_text):
    """Calculates the richness of the vocabulary used in a full piece of text with stopwords included"""

    # Remove punctuation and turn text into lower case
    processed_input_text = pp.TextPreprocessor.remove_punctuation(input_text).lower()

    # Tokenize the text into individual words
    words = nltk.word_tokenize(processed_input_text)
    # print(f'Words : {words}')
    # print(f'Num of words : {len(words)}')

    # Calculate the total number of words
    total_words = len(words)

    # Calculate the total number of unique words
    unique_words = len(set(words))

    # Calculate the vocabulary richness
    vocabulary_richness = unique_words / total_words

    return vocabulary_richness


def calculate_total_vocab_richness_no_stopwords(input_text):
    """Calculates the richness of the vocabulary used in a full piece of text with stopwords removed"""

    # Remove punctuation and turn text into lower case
    processed_input_text = pp.TextPreprocessor.remove_punctuation(input_text).lower()

    stop_words = set(stopwords.words('english'))

    # Tokenize the text into individual words
    words = nltk.word_tokenize(processed_input_text)
    # print(f'Words : {words}')
    # print(f'Num of words : {len(words)}')

    # Filtered sentence without stopwords
    filtered_sentences = [w for w in words if not w in stop_words]
    # print(f'Filtered Words : {filtered_sentences}')
    # print(f'Num of words : {len(filtered_sentences)}')

    # Calculate the total number of words
    total_words = len(filtered_sentences)

    # Calculate the total number of unique words
    unique_words = len(set(filtered_sentences))

    # Calculate the vocabulary richness
    vocabulary_richness_no_stopwords = unique_words / total_words

    return vocabulary_richness_no_stopwords


print('HUMAN WRITTEN TEXT')
print(f'Vocab richness of text (including stopwords) : {calculate_total_vocab_richness(human_text_to_analyse)}')
print(f'Vocab richness of text (excluding stopwords) : {calculate_total_vocab_richness_no_stopwords(human_text_to_analyse)}')

print('AI WRITTEN TEXT')
print(f'Vocab richness of text (including stopwords) : {calculate_total_vocab_richness(ai_text_to_analyse)}')
print(f'Vocab richness of text (excluding stopwords) : {calculate_total_vocab_richness_no_stopwords(ai_text_to_analyse)}')