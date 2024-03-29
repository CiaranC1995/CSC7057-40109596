import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import datetime

start_time = time.time()

# Read in dataset from csv file
dataset = pd.read_csv(
    r"C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro\80percent_of_original_dataset.csv")

print(f'Dataset Read @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

final_text = dataset['text_to_analyse'].to_list()
final_ppl = dataset['perplexity'].to_list()
final_burstiness = dataset['burstiness'].to_list()
final_labels = dataset['AI Generated'].to_list()

# 80% Train / 20% Validation split
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

# Model input array creation
final_x_train = np.hstack((X_train_text_vectorized.toarray(), np.array(X_train_perplexity).reshape(-1, 1), np.array(X_train_burstiness).reshape(-1, 1)))
final_x_test = np.hstack((X_test_text_vectorized.toarray(), np.array(X_test_perplexity).reshape(-1, 1), np.array(X_test_burstiness).reshape(-1, 1)))

print(f'Final Train and Test Data Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# SVM Model Implementation
svm_model = LinearSVC()

# Hyperparameter grid for grid search
param_grid = {
    'loss': ['squared_hinge'],
    'max_iter': [10000, 20000],
    'dual': [False],
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
}

# Necessary to split into smaller batches to combat large memory and computational resource overhead
batch_size = 5000

# Split the training data into smaller batches
num_samples = final_x_train.shape[0]
num_batches = (num_samples + batch_size - 1) // batch_size

# 10 fold
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_samples)

    # Get the current batch of training data
    batch_x_train = final_x_train[start_idx:end_idx]
    batch_y_train = y_train[start_idx:end_idx]
    print(f'Starting Grid Search @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

    # Perform grid search using cross-validation on the current batch
    grid_search.fit(batch_x_train, batch_y_train)

# Get the best SVM model
best_svm_model = grid_search.best_estimator_

# Print the best hyperparameters found during grid search
print(f"Best Hyperparameters:")
print(grid_search.best_params_)

# Train the model on the entire training set using the best hyperparameters
best_svm_model.fit(final_x_train, y_train)

# Perform cross-validation using the best model
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(best_svm_model, final_x_train, y_train, cv=5, scoring=scoring_metrics)

# Access the cross-validation results for each metric
cv_accuracy = cv_results['test_accuracy']
cv_precision = cv_results['test_precision']
cv_recall = cv_results['test_recall']
cv_f1 = cv_results['test_f1']
cv_roc_auc = cv_results['test_roc_auc']

# Print mean and standard deviation for each metric
print("Cross-validation Accuracy:", np.mean(cv_accuracy), "+/-", np.std(cv_accuracy))
print("Cross-validation Precision:", np.mean(cv_precision), "+/-", np.std(cv_precision))
print("Cross-validation Recall:", np.mean(cv_recall), "+/-", np.std(cv_recall))
print("Cross-validation F1-score:", np.mean(cv_f1), "+/-", np.std(cv_f1))
print("Cross-validation AUC-ROC:", np.mean(cv_roc_auc), "+/-", np.std(cv_roc_auc))

# Best model prediction on validation subset
y_pred_svm = best_svm_model.predict(final_x_test)
pred_probabilities = best_svm_model._predict_proba_lr(final_x_test)

# Find misclassified record indices
misclassified_indices = np.where(y_test != y_pred_svm)[0]

# Collect misclassified records and their associated values
misclassified_data = []
for index in misclassified_indices:
    record = {
        'text': X_test_text[index],
        'true_label': y_test[index],
        'predicted_label': y_pred_svm[index],
        'prediction_probability': pred_probabilities[index][int(y_pred_svm[index])],
        'perplexity': X_test_perplexity[index],
        'burstiness': X_test_burstiness[index]
    }
    misclassified_data.append(record)

# Convert the collected data to a DataFrame
misclassified_df = pd.DataFrame(misclassified_data)

# Save misclassified records to a CSV file
csv_output_file = r"C:\Users\ccase\Desktop\Dissertation\cross_validated_misclassified_records.csv"
misclassified_df.to_csv(csv_output_file, index=False)

# Find correctly classified record indices
correctly_classified_indices = np.where(y_test == y_pred_svm)[0]

# Collect correctly classified records and their associated values
correctly_classified_data = []
for index in correctly_classified_indices:
    record = {
        'text': X_test_text[index],
        'true_label': y_test[index],
        'predicted_label': y_pred_svm[index],
        'prediction_probability': pred_probabilities[index][int(y_pred_svm[index])],
        'perplexity': X_test_perplexity[index],
        'burstiness': X_test_burstiness[index]
    }
    correctly_classified_data.append(record)

# Convert the collected data to a DataFrame
correctly_classified_df = pd.DataFrame(correctly_classified_data)

# Save records to a CSV file
csv_output_file = r"C:\Users\ccase\Desktop\Dissertation\cross_validated_correctly_classified_records.csv"
correctly_classified_df.to_csv(csv_output_file, index=False)

print(f'Model Test Data Predictions Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# Save model for future deployment
model_path = r'./Models/LinearSVC_CV.pickle'

with open(model_path, 'wb') as file:
    pickle.dump(best_svm_model, file)

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
