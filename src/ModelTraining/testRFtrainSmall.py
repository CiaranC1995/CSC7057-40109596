import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import datetime

start_time = time.time()

# Read in dataset from csv file
dataset = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets"
                      r"\detokenized_preprocessed_dataset_noTransformation.csv")

print(f'Dataset Read @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# Separate the records with attribute 0 and 1
human_written_examples = dataset[dataset['AI Generated'] == 0].sample(n=5000, random_state=42)
ai_generated_examples = dataset[dataset['AI Generated'] == 1].sample(n=5000, random_state=42)

# Concatenate the two samples into a new DataFrame
new_df = pd.concat([human_written_examples, ai_generated_examples], ignore_index=True)

final_text = new_df['text_to_analyse'].to_list()
final_ppl = new_df['perplexity'].to_list()
final_burstiness = new_df['burstiness'].to_list()
final_labels = new_df['AI Generated'].to_list()

X_train_text, X_test_text, X_train_perplexity, X_test_perplexity, X_train_burstiness, X_test_burstiness, y_train, y_test = train_test_split(
    final_text, final_ppl, final_burstiness, final_labels, test_size=0.2, random_state=42
)

print(f'Dataset Split @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')
print('Size of Training Set :', len(X_train_text))

# Vectorize the textual inputs
vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(dataset['text_to_analyse'])

X_train_text_vectorized = vectorizer.transform(X_train_text)
X_test_text_vectorized = vectorizer.transform(X_test_text)

vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'

with open(vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)

print(f'Vocabulary Fit Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

final_x_train = np.hstack((X_train_text_vectorized.toarray(), np.array(X_train_perplexity).reshape(-1, 1), np.array(X_train_burstiness).reshape(-1, 1)))
final_x_test = np.hstack((X_test_text_vectorized.toarray(), np.array(X_test_perplexity).reshape(-1, 1), np.array(X_test_burstiness).reshape(-1, 1)))

print(f'Final Train and Test Data Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(final_x_train, y_train)

print(f'Model Fit Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

y_pred_rf = rf_model.predict(final_x_test)

print(f'Model Test Data Predictions Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {(rf_accuracy * 100):.3f}%")
print(f"Random Forest Precision: {rf_precision:.3f}")
print(f"Random Forest Recall: {rf_recall:.3f}")
print(f"Random Forest F1-score: {rf_f1:.3f}")
print(f"Random Forest AUC-ROC: {rf_roc_auc:.3f}")

model_path = r'./Models/RF_Classifier_EntireDataset_detokenized_NoTransformationOfPPL_Burst.pickle'

with open(model_path, 'wb') as file:
    pickle.dump(rf_model, file)

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["AI Generated", "Human Written"])
disp.plot()
plt.title("Random Forest Model Confusion Matrix")
plt.show()

# Random Forest AUC-ROC Curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()

print("Time Elapsed: {:.2f}s".format(time.time() - start_time))
print("Program finished executing at:", datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))
