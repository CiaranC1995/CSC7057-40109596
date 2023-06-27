import datetime
import time

import pandas as pd
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer, TreebankWordDetokenizer, sent_tokenize
from nltk.corpus import stopwords
from scipy.stats import boxcox, yeojohnson

start_time = time.time()

dataset = pd.read_csv("dataset_ready_to_split_and_transform.csv")

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocess text
preprocessed_texts = [
    TreebankWordDetokenizer().detokenize(
        [
            lemmatizer.lemmatize(stemmer.stem(word))
            for word in word_tokenize(sentence.lower())
            if word not in stopwords.words('english')
        ]
    )
    for text in dataset['text_to_analyse']
    for sentence in sent_tokenize(text)
]

processed_dataframe_boxcox = pd.DataFrame()
processed_dataframe_yeojohnson = pd.DataFrame()

processed_dataframe_boxcox['text_to_analyse'] = preprocessed_texts
processed_dataframe_yeojohnson['text_to_analyse'] = preprocessed_texts

transformed_perplexity_boxcox, _ = boxcox(dataset['perplexity'])
transformed_burstiness_boxcox, _ = boxcox(dataset['burstiness'])

transformed_perplexity_yeojohnson, _ = yeojohnson(dataset['perplexity'])
transformed_burstiness_yeojohnson, _ = yeojohnson(dataset['burstiness'])

processed_dataframe_boxcox['perplexity'] = transformed_perplexity_boxcox
processed_dataframe_boxcox['burstiness'] = transformed_burstiness_boxcox
processed_dataframe_yeojohnson['perplexity'] = transformed_perplexity_yeojohnson
processed_dataframe_yeojohnson['burstiness'] = transformed_burstiness_yeojohnson

processed_dataframe_boxcox['AI Generated'] = dataset['AI Generated']
processed_dataframe_yeojohnson['AI Generated'] = dataset['AI Generated']

processed_dataframe_boxcox = processed_dataframe_boxcox.sample(frac=1).reset_index(drop=True)
processed_dataframe_yeojohnson = processed_dataframe_yeojohnson.sample(frac=1).reset_index(drop=True)

processed_dataframe_boxcox.to_csv('original_preprocessing_method_dataset_boxcox.csv', index=False)
processed_dataframe_yeojohnson.to_csv('original_preprocessing_method_dataset_yeojohnson.csv', index=False)

print("\nTime Elapsed: {:.2f}s".format(time.time() - start_time))
print("Program finished executing at:", datetime.datetime.now())
