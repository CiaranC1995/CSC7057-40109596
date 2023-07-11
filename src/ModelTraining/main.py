import pickle
import numpy as np
from src.ClassificationMetrics.perplexity import PerplexityBurstiness as pb
from collections import defaultdict
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

word_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()
model_path = r'./Models/LinearSVC_Classifier_EntireDataset_detokenized_NoTransformationOfPPL_Burst.pickle'
vect_path = r'./Vectorizers/tfidfvectorizer.pickle'


def process_entry(entry):
    """Lemmatize entry and remove stopwords"""
    words = [
        word_lemmatizer.lemmatize(word, tag_map.get(tag[0], 'n'))
        for word, tag in pos_tag(entry)
        if word.isalpha() and word not in stop_words
    ]
    return ' '.join(words)


def classify_text(input_text, model_path, vectorizer_path):

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

    lower_case_text = input_text.lower()

    tokenized_text = word_tokenize(lower_case_text)

    preprocessed_text = process_entry(tokenized_text)

    results = pb.process_text_ppl_burstiness(input_text)
    perplexity_value = results['avg_text_ppl']
    burstiness_value = results["text_burstiness"]

    # vectorize the text to be analysed
    vectorized_text = vectorizer.transform([preprocessed_text])

    text_array = vectorized_text.toarray()
    ppl_array = np.array(perplexity_value).reshape(-1, 1)
    burst_array = np.array(burstiness_value).reshape(-1, 1)

    ppl_array = np.repeat(ppl_array, text_array.shape[0], axis=0)
    burst_array = np.repeat(burst_array, text_array.shape[0], axis=0)

    # inside an array pass the vectorized text, perplexity, burstiness
    model_input = np.hstack((text_array, ppl_array, burst_array))
    # predict using the model the classification
    predicted_output = model.predict(model_input)
    pred_probabilities = model._predict_proba_lr(model_input)
    human_probability = pred_probabilities[0][0]
    ai_probability = pred_probabilities[0][1]

    return {
        'processed_input_text': preprocessed_text,
        'text_perplexity': perplexity_value,
        'text_burstiness': burstiness_value,
        'classifier_output': predicted_output,
        'human_probability': human_probability,
        'ai_probability': ai_probability
    }
