import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import time
import datetime

start_time = time.time()

# Read in dataset from csv file
dataset = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\preprocessed_dataset_boxcox.csv")

vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'

with open(vectorizer_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

print(f'Dataset Read @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

final_text = dataset['text_to_analyse'].to_list()
final_ppl = dataset['perplexity'].to_list()
final_burstiness = dataset['burstiness'].to_list()
final_labels = dataset['AI Generated'].to_list()

X_train_text, X_test_text, X_train_perplexity, X_test_perplexity, X_train_burstiness, X_test_burstiness, y_train, y_test = train_test_split(
    final_text, final_ppl, final_burstiness, final_labels, test_size=0.2, random_state=42
)

print(f'Dataset Split @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')
print('Size of Training Set :', len(X_train_text))

X_train_text_vectorized = vectorizer.transform(X_train_text)
X_test_text_vectorized = vectorizer.transform(X_test_text)

print(f'Text Vectorization Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

final_x_train = np.hstack((X_train_text_vectorized.toarray(), np.array(X_train_perplexity).reshape(-1, 1), np.array(X_train_burstiness).reshape(-1, 1)))
final_x_test = np.hstack((X_test_text_vectorized.toarray(), np.array(X_test_perplexity).reshape(-1, 1), np.array(X_test_burstiness).reshape(-1, 1)))

print(f'Final Train and Test Data Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

parameter_grid = {
    'C': [0.1, 1, 10],
    'loss': ['squared_hinge'],
    'max_iter': [1000, 10000],
    'dual': [True, False]
}

# SVM
svm_model = LinearSVC()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=parameter_grid, cv=5, scoring='accuracy')
grid_search.fit(final_x_train, y_train)

# Get the best model and its parameters
best_svm_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Train the best SVM model
best_svm_model.fit(final_x_train, y_train)

print(f'Model Fit Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# Test the best model
y_pred_svm = best_svm_model.predict(final_x_test)

print(f'Model Test Data Predictions Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_roc_auc = roc_auc_score(y_test, y_pred_svm)

print(f"SVM Accuracy: {svm_accuracy * 100}%")
print(f"SVM Precision: {svm_precision:.3f}")
print(f"SVM Recall: {svm_recall:.3f}")
print(f"SVM F1-score: {svm_f1:.3f}")
print(f"SVM AUC-ROC: {svm_roc_auc:.3f}")

model_path = r'./Model/Best_SVM_Classifier_Pipeline.pickle'

with open(model_path, 'wb') as file:
    pickle.dump(best_svm_model, file)

num_weights_for_input_feats = np.count_nonzero(best_svm_model.coef_)
print("Weights assigned to each feature in the input data:", num_weights_for_input_feats)

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["AI Generated", "Human Written"])
disp.plot()
plt.title("SVM Model Confusion Matrix")
plt.show()

# SVM AUC-ROC Curve
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_svm)
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

print("Time Elapsed: {:.2f}s".format(time.time() - start_time))
print("Program finished executing at:", datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))