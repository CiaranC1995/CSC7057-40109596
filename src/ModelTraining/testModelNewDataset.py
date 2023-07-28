import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from nltk import WordNetLemmatizer, pos_tag
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_score, roc_auc_score, roc_curve, auc
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

model_path = r'C:\Users\ccase\Desktop\CSC7057-40109596\src\ModelTraining\Models' \
             r'\LinearSVC_CV.pickle'

test_data = r'C:\Users\ccase\Desktop\Dissertation\Datasets\chatgpt-generated-text-corpus' \
            r'\NEW_DATASET_perplexity_burstiness_scores.csv'

vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'


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


with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

dataset = pd.read_csv(test_data)
dataset.drop('Filename', axis=1, inplace=True)

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

# dataset.to_csv("dataset_with_preprocessed_text.csv")
dataset.drop('text_to_analyse', axis=1, inplace=True)

dataset = dataset[['text_final', 'perplexity', 'burstiness', 'AI_Generated']]

new_text = dataset['text_final'].to_list()
new_text_vectorized = vectorizer.transform(new_text)

new_ppl = dataset['perplexity'].to_list()
new_burstiness = dataset['burstiness'].to_list()
new_x = np.hstack((new_text_vectorized.toarray(), np.array(new_ppl).reshape(-1, 1), np.array(new_burstiness).reshape(-1, 1)))

new_labels_pred = classifier.predict(new_x)

new_labels_true = dataset['AI_Generated'].to_list()

svm_accuracy = accuracy_score(new_labels_true, new_labels_pred)
svm_precision = precision_score(new_labels_true, new_labels_pred)
svm_recall = recall_score(new_labels_true, new_labels_pred)
svm_f1 = f1_score(new_labels_true, new_labels_pred)
svm_roc_auc = roc_auc_score(new_labels_true, new_labels_pred)

print(f"SVM Accuracy: {(svm_accuracy * 100):.3f}%")
print(f"SVM Precision: {svm_precision:.3f}")
print(f"SVM Recall: {svm_recall:.3f}")
print(f"SVM F1-score: {svm_f1:.3f}")
print(f"SVM AUC-ROC: {svm_roc_auc:.3f}")

# SVM Confusion Matrix
cm_svm = confusion_matrix(new_labels_true, new_labels_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["AI Generated", "Human Written"])
disp.plot()
plt.title("SVM Model Confusion Matrix")
plt.show()

# SVM AUC-ROC Curve
fpr_svm, tpr_svm, thresholds_svm = roc_curve(new_labels_true, new_labels_pred)
roc_auc = auc(fpr_svm, tpr_svm)
plt.figure()
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
plt.legend(loc="lower right")
plt.show()


