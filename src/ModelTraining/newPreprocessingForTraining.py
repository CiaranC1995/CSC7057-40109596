import datetime
import time
from collections import defaultdict
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import numpy as np
from scipy.stats import boxcox, yeojohnson

start_time = time.time()


def process_entry(entry):
    """Lemmatize entry and remove stopwords"""
    word_lemmatizer = WordNetLemmatizer()
    words = [
        word_lemmatizer.lemmatize(word, tag_map.get(tag[0], 'n'))
        for word, tag in pos_tag(entry)
        if word.isalpha() and word not in stop_words
    ]
    final_words = ' '.join(words)
    return final_words


dataset = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\dataset_ready_to_split_and_transform.csv")

# Remove blank rows if any
dataset['text_to_analyse'].dropna(inplace=True)
# Change all the text to lower case
dataset['text_to_analyse'] = [entry.lower() for entry in dataset['text_to_analyse']]
# Tokenization
dataset['text_to_analyse'] = [word_tokenize(record) for record in dataset['text_to_analyse']]
# Remove Stop words, Non-Numeric and perform Word Stemming/Lemmatization.
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

stop_words = set(stopwords.words('english'))

# Apply the processing function to each entry in 'text_to_analyse' column
dataset['text_final'] = dataset['text_to_analyse'].apply(process_entry)

# new dataframes to store processed data columns
# processed_dataframe_boxcox = pd.DataFrame()
# processed_dataframe_yeojohnson = pd.DataFrame()
processed_dataframe_logarithmic = pd.DataFrame()

# dataframe to hold non-transformed ppl and burstiness values
# processed_dataframe = pd.DataFrame()
# processed_dataframe['text_to_analyse'] = dataset['text_final']
# processed_dataframe['perplexity'] = dataset['perplexity']
# processed_dataframe['burstiness'] = dataset['burstiness']
# processed_dataframe['AI Generated'] = dataset['AI Generated']
# processed_dataframe = processed_dataframe.sample(frac=1).reset_index(drop=True)
# processed_dataframe.to_csv('preprocessed_dataset_noTransformation.csv', index=False)

# populate with preprocessed text
# processed_dataframe_boxcox['text_to_analyse'] = dataset['text_final']
# processed_dataframe_yeojohnson['text_to_analyse'] = dataset['text_final']
processed_dataframe_logarithmic['text_to_analyse'] = dataset['text_final']

# transform ppl and burstiness using boxcox (no negative values)
# transformed_perplexity_boxcox, _ = boxcox(dataset['perplexity'])
# transformed_burstiness_boxcox, _ = boxcox(dataset['burstiness'])

# transform using yeojohnson (allows negative values)
# transformed_perplexity_yeojohnson, _ = yeojohnson(dataset['perplexity'])
# transformed_burstiness_yeojohnson, _ = yeojohnson(dataset['burstiness'])

# transform using logarithmic method
transformed_perplexity_logarithmic = np.log(dataset['perplexity'])
transformed_burstiness_logarithmic = np.log(dataset['burstiness'])

# populate dataframes with transformed values
# processed_dataframe_boxcox['perplexity'] = transformed_perplexity_boxcox
# processed_dataframe_boxcox['burstiness'] = transformed_burstiness_boxcox
# processed_dataframe_yeojohnson['perplexity'] = transformed_perplexity_yeojohnson
# processed_dataframe_yeojohnson['burstiness'] = transformed_burstiness_yeojohnson
processed_dataframe_logarithmic['perplexity'] = transformed_perplexity_logarithmic
processed_dataframe_logarithmic['burstiness'] = transformed_burstiness_logarithmic

# populate dataframes with labels
# processed_dataframe_boxcox['AI Generated'] = dataset['AI Generated']
# processed_dataframe_yeojohnson['AI Generated'] = dataset['AI Generated']
processed_dataframe_logarithmic['AI Generated'] = dataset['AI Generated']

# randomise dataframes
# processed_dataframe_boxcox = processed_dataframe_boxcox.sample(frac=1).reset_index(drop=True)
# processed_dataframe_yeojohnson = processed_dataframe_yeojohnson.sample(frac=1).reset_index(drop=True)
processed_dataframe_logarithmic = processed_dataframe_logarithmic.sample(frac=1).reset_index(drop=True)

# output to csv files
# processed_dataframe_boxcox.to_csv('preprocessed_dataset_boxcox.csv', index=False)
# processed_dataframe_yeojohnson.to_csv('preprocessed_dataset_yeojohnson.csv', index=False)
processed_dataframe_logarithmic.to_csv('preprocessed_dataset_logarithmic.csv', index=False)

print("\nTime Elapsed: {:.2f}s".format(time.time() - start_time))
print("Program finished executing at:", datetime.datetime.now())
