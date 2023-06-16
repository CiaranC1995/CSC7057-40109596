import nltk
import numpy as np
import pandas as pd
from numpy import hstack
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from src.ClassificationMetrics import perplexity as pp
from src.ClassificationMetrics.TextPreprocessor import preprocessor as prep

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Read in dataset from csv file
dataset = pd.read_csv("test.csv")

dataframe = pd.DataFrame()

# Populating dataframe with features to be used for training model
dataframe['text_to_analyse'] = dataset['Text']
dataframe["perplexity"] = [pp.PerplexityBurstiness.process_text_ppl_burstiness(text)['avg_text_ppl'] for text in
                           dataframe["text_to_analyse"]]
dataframe["burstiness"] = [pp.PerplexityBurstiness.process_text_ppl_burstiness(text)['text_burstiness'] for text in
                           dataframe["text_to_analyse"]]
dataframe["AI Generated"] = dataset["AI-Generated"]

# Drop the "text_to_analyse" column
dataframe.drop("text_to_analyse", axis=1, inplace=True)

# Exporting the dataframe to a CSV file
dataframe.to_csv("output.csv", index=True)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(dataframe.index, dataframe["perplexity"], label="Perplexity")
plt.xlabel("Index")
plt.ylabel("Perplexity")
plt.title("Perplexity Plot")

ai_generated_markers = ['r' if val == 1 else 'b' for val in dataframe["AI Generated"]]
plt.scatter(dataframe.index, dataframe["perplexity"], c=ai_generated_markers, label="AI Generated")

# Plotting burstiness
plt.subplot(2, 1, 2)
plt.plot(dataframe.index, dataframe["burstiness"], label="Burstiness")
plt.xlabel("Index")
plt.ylabel("Burstiness")
plt.title("Burstiness Plot")

# Adding AI Generated column as color-coded markers
ai_generated_markers = ['r' if val == 1 else 'b' for val in dataframe["AI Generated"]]
plt.scatter(dataframe.index, dataframe["burstiness"], c=ai_generated_markers, label="AI Generated")


# Adding legend
plt.legend()

# Adjusting layout
plt.tight_layout()

# Displaying the plot
plt.show()

# # Perform NLP techniques
# lowercase_texts = [word.lower() for word in dataframe['text_to_analyse']]
# # print(lowercase_texts)
# # print(len(lowercase_texts))
# segmented_texts = [prep.TextPreprocessor.split_into_sentences(text) for text in lowercase_texts]
# # print(segmented_texts)
# # print(len(segmented_texts))
# tokenized_texts = [[word_tokenize(sent) for sent in sents] for sents in segmented_texts]
# # print(tokenized_texts)
# # print(len(tokenized_texts))
# stopwords_removed_texts = [[[word for word in sent if word not in stopwords.words('english')] for sent in text] for text
#                            in tokenized_texts]
# # print(stopwords_removed_texts)
# # print(len(stopwords_removed_texts))
#
# # Stemming
# stemmer = PorterStemmer()
# stemmed_texts = [[[stemmer.stem(word) for word in sent] for sent in text] for text in stopwords_removed_texts]
# # print(stemmed_texts)
# # print(len(stemmed_texts))
#
# # Lemmatization
# lemmatizer = WordNetLemmatizer()
# lemmatized_texts = [[[lemmatizer.lemmatize(word) for word in sent] for sent in text] for text in stemmed_texts]
# # print(lemmatized_texts)
# # print(len(lemmatized_texts))
#
# detokenized_text = [[TreebankWordDetokenizer().detokenize(sent) for sent in text] for text in lemmatized_texts]
# # print(detokenized_text)
# # print(len(detokenized_text))
#
# unsegmented_text = [TreebankWordDetokenizer().detokenize(sents) for sents in detokenized_text]
# # print(unsegmented_text)
# # print(len(unsegmented_text))
#
# # adding preprocessed text to the dataframe
# dataframe['text_to_analyse'] = unsegmented_text
#
# # Separate the features and the labels
# # features = dataframe[['text_to_analyse', 'perplexity', 'burstiness']]
# # labels = dataframe['AI Generated']
#
# final_text = dataframe['text_to_analyse'].to_list()
# final_ppl = dataframe['perplexity'].to_list()
# final_burstiness = dataframe['burstiness'].to_list()
#
# final_labels = dataframe['AI Generated'].to_list()
#
# X_train_text, X_test_text, X_train_perplexity, X_test_perplexity, X_train_burstiness, X_test_burstiness, y_train, y_test = train_test_split(
#     final_text, final_ppl, final_burstiness, final_labels, test_size=0.2, random_state=42
# )
#
# # Vectorize the textual inputs
# vectorizer = TfidfVectorizer()
# X_train_text_vectorized = vectorizer.fit_transform(X_train_text).toarray()
# X_test_text_vectorized = vectorizer.transform(X_test_text).toarray()
#
# X_train_perplexity = np.array(X_train_perplexity).reshape(-1, 1)
# X_test_perplexity = np.array(X_test_perplexity).reshape(-1, 1)
# X_train_burstiness = np.array(X_train_burstiness).reshape(-1, 1)
# X_test_burstiness = np.array(X_test_burstiness).reshape(-1, 1)
#
# # Combine the textual inputs with the additional features
# X_train = hstack((X_train_text_vectorized, X_train_perplexity, X_train_burstiness))
# X_test = hstack((X_test_text_vectorized, X_test_perplexity, X_test_burstiness))
#
# # Train the SVM model
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
#
# # Evaluate the model
# accuracy = svm_model.score(X_test, y_test)
# print("Accuracy:", accuracy)
#
# predictions = svm_model.predict(X_test)
#
# print("-------------CONFUSION MATRIX FOR CLASSIFIER-------------")
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AI Generated", "Human Written"])
# disp.plot()
# plt.show()
