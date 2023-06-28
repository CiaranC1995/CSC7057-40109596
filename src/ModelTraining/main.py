import time
import datetime
import pickle
import numpy as np
from src.ClassificationMetrics.perplexity import PerplexityBurstiness as pb
from collections import defaultdict
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag

start_time = time.time()


def process_entry(entry):
    """Lemmatize entry and remove stopwords"""
    final_words = [
        word_lemmatizer.lemmatize(word, tag_map.get(tag[0], 'n'))
        for word, tag in pos_tag(entry)
        if word.isalpha() and word not in stop_words
    ]
    return str(final_words)


model_path = r'./Models/SVM_Classifier_EntireDataset_DetokenizedTextTrain.pickle'
vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_CiaranSoftwareEngineer.txt') as f:
    input_text = f.read()

lower_case_text = input_text.lower()

tokenized_text = word_tokenize(lower_case_text)

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

word_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

preprocessed_text = process_entry(tokenized_text)

results = pb.process_text_ppl_burstiness(input_text)
perplexity_value = results['avg_text_ppl']
burstiness_value = results["text_burstiness"]

print(f'PPL : {perplexity_value} and Burstiness : {burstiness_value}')

transformed_perplexity_logarithmic = np.log(perplexity_value)
transformed_burstiness_logarithmic = np.log(burstiness_value)

text_to_vec = ' '.join(eval(preprocessed_text))

# vectorize the text to be analysed
vectorized_text = vectorizer.transform([text_to_vec])

text_array = vectorized_text.toarray()
ppl_array = np.array(transformed_perplexity_logarithmic).reshape(-1, 1)
burst_array = np.array(transformed_burstiness_logarithmic).reshape(-1, 1)

ppl_array = np.repeat(ppl_array, text_array.shape[0], axis=0)
burst_array = np.repeat(burst_array, text_array.shape[0], axis=0)

# inside an array pass the vectorized text, perplexity, burstiness
model_input = np.hstack((text_array, ppl_array, burst_array))
# predict using the model the classification
predicted_output = model.predict(model_input)
pred_probabilities = model._predict_proba_lr(model_input)

human_probability = pred_probabilities[0][0]
ai_probability = pred_probabilities[0][1]

if predicted_output[0] == 0:
    print(f'There is a {(human_probability * 100):.2f}% chance that this text was written by a human.')
else:
    print(f'There is a {(ai_probability * 100):.2f}% chance that this text was generated by AI.')


print("Time Elapsed: {:.2f}s".format(time.time() - start_time))
print("Program finished executing at:", datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))