import pickle
import numpy as np
from classifierFiles.perplexity import PerplexityBurstiness as pb
from collections import defaultdict
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag

class TextClassifier:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.tag_map = defaultdict(lambda: wn.NOUN)
        self.tag_map['J'] = wn.ADJ
        self.tag_map['V'] = wn.VERB
        self.tag_map['R'] = wn.ADV
        self.word_lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def process_entry(self, entry):
        """Lemmatize entry and remove stopwords"""
        words = [
            self.word_lemmatizer.lemmatize(word, self.tag_map.get(tag[0], 'n'))
            for word, tag in pos_tag(entry)
            if word.isalpha() and word not in self.stop_words
        ]
        return ' '.join(words)
    
    def load_model(self):
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)
    
    def load_vectorizer(self):
        with open(self.vectorizer_path, 'rb') as vec_file:
            self.vectorizer = pickle.load(vec_file)
    
    def classify_text(self, input_text):
        if self.model is None:
            self.load_model()
        if self.vectorizer is None:
            self.load_vectorizer()
        
        lower_case_text = input_text.lower()
        tokenized_text = word_tokenize(lower_case_text)
        preprocessed_text = self.process_entry(tokenized_text)
        
        results = pb.process_text_ppl_burstiness(input_text)
        perplexity_value = results['avg_text_ppl']
        burstiness_value = results["text_burstiness"]

        if np.isnan(perplexity_value) or np.isnan(burstiness_value):
            return {
                'error_message': 'Error. Classification Unsuccessful. Please try again.'
            }

        vectorized_text = self.vectorizer.transform([preprocessed_text])
        text_array = vectorized_text.toarray()
        ppl_array = np.array(perplexity_value).reshape(-1, 1)
        burst_array = np.array(burstiness_value).reshape(-1, 1)

        ppl_array = np.repeat(ppl_array, text_array.shape[0], axis=0)
        burst_array = np.repeat(burst_array, text_array.shape[0], axis=0)

        model_input = np.hstack((text_array, ppl_array, burst_array))
        predicted_output = self.model.predict(model_input)
        pred_probabilities = self.model._predict_proba_lr(model_input)
        human_probability = pred_probabilities[0][0]
        ai_probability = pred_probabilities[0][1]

        return {
            'processed_input_text': preprocessed_text,
            'text_perplexity': perplexity_value,
            'text_burstiness': burstiness_value,
            'classifier_output': predicted_output,
            'human_probability': human_probability,
            'ai_probability': ai_probability,
            'sentences': results['sentences'],
            'sentence_perplexities': results['sentence_perplexities']
        }
