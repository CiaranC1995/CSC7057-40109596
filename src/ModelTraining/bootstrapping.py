import pickle
import numpy as np
import pandas as pd
from seaborn.algorithms import bootstrap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import datetime
import seaborn as sns


# Function to evaluate SVM model and calculate performance metrics
def evaluate_svm_model(X_train, X_test, y_train, y_test):
    svm_model = LinearSVC(loss='squared_hinge', max_iter=10000, dual=False)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_recall = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)
    svm_roc_auc = roc_auc_score(y_test, y_pred_svm)

    return svm_accuracy, svm_precision, svm_recall, svm_f1, svm_roc_auc


# Read in dataset from csv file
dataset = pd.read_csv(
    r"C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro\detokenized_preprocessed_dataset_noTransformation.csv")

final_text = dataset['text_to_analyse'].to_list()
final_ppl = dataset['perplexity'].to_list()
final_burstiness = dataset['burstiness'].to_list()
final_labels = dataset['AI Generated'].to_list()

X_train_text, X_test_text, X_train_perplexity, X_test_perplexity, X_train_burstiness, X_test_burstiness, y_train, y_test = train_test_split(
    final_text, final_ppl, final_burstiness, final_labels, test_size=0.2, random_state=42
)

vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

X_train_text_vectorized = vectorizer.transform(X_train_text)
X_test_text_vectorized = vectorizer.transform(X_test_text)


final_x_train = np.hstack((X_train_text_vectorized.toarray(), np.array(X_train_perplexity).reshape(-1, 1),
                           np.array(X_train_burstiness).reshape(-1, 1)))
final_x_test = np.hstack((X_test_text_vectorized.toarray(), np.array(X_test_perplexity).reshape(-1, 1),
                          np.array(X_test_burstiness).reshape(-1, 1)))

# Bootstrapping for 1000 iterations
batch_size = 5000
num_iterations = 1000
svm_accuracy_list = []
svm_precision_list = []
svm_recall_list = []
svm_f1_list = []
svm_roc_auc_list = []

print(f'Beginning Iterations @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

for iteration_count, i in enumerate(range(num_iterations)):
    # Bootstrap sample indices
    sample_indices = np.random.choice(range(len(X_train_text)), size=len(X_train_text), replace=True)

    # Divide the bootstrap sample indices into batches
    num_batches = len(sample_indices) // batch_size
    batches = np.array_split(sample_indices, num_batches)

    # Initialize lists to store metrics for each batch
    batch_accuracy = []
    batch_precision = []
    batch_recall = []
    batch_f1 = []
    batch_auc = []

    for batch_count, batch_indices in enumerate(batches):
        # Generate bootstrap samples for the current batch
        X_train_bootstrap = final_x_train[batch_indices]
        y_train_bootstrap = np.array(y_train)[batch_indices]

        # Evaluate SVM model on the current batch
        acc, prec, rec, f1, auc = evaluate_svm_model(X_train_bootstrap, final_x_test, y_train_bootstrap, y_test)

        # Append batch metrics to corresponding lists
        batch_accuracy.append(acc)
        batch_precision.append(prec)
        batch_recall.append(rec)
        batch_f1.append(f1)
        batch_auc.append(auc)

        # Print the count during each batch iteration
        print(f'Iteration {iteration_count + 1}/{num_iterations}, Batch {batch_count + 1}/{num_batches} Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')
        print()

    # Append mean of batch metrics to the overall lists
    svm_accuracy_list.append(np.mean(batch_accuracy))
    svm_precision_list.append(np.mean(batch_precision))
    svm_recall_list.append(np.mean(batch_recall))
    svm_f1_list.append(np.mean(batch_f1))
    svm_roc_auc_list.append(np.mean(batch_auc))

# Calculate the mean and confidence intervals for each performance metric
svm_accuracy_mean = np.mean(svm_accuracy_list)
svm_accuracy_ci = np.percentile(svm_accuracy_list, [2.5, 97.5])

svm_precision_mean = np.mean(svm_precision_list)
svm_precision_ci = np.percentile(svm_precision_list, [2.5, 97.5])

svm_recall_mean = np.mean(svm_recall_list)
svm_recall_ci = np.percentile(svm_recall_list, [2.5, 97.5])

svm_f1_mean = np.mean(svm_f1_list)
svm_f1_ci = np.percentile(svm_f1_list, [2.5, 97.5])

svm_roc_auc_mean = np.mean(svm_roc_auc_list)
svm_roc_auc_ci = np.percentile(svm_roc_auc_list, [2.5, 97.5])

print(f"SVM Accuracy: Mean={svm_accuracy_mean:.3f}, CI=({svm_accuracy_ci[0]:.3f}, {svm_accuracy_ci[1]:.3f})")
print(f"SVM Precision: Mean={svm_precision_mean:.3f}, CI=({svm_precision_ci[0]:.3f}, {svm_precision_ci[1]:.3f})")
print(f"SVM Recall: Mean={svm_recall_mean:.3f}, CI=({svm_recall_ci[0]:.3f}, {svm_recall_ci[1]:.3f})")
print(f"SVM F1-score: Mean={svm_f1_mean:.3f}, CI=({svm_f1_ci[0]:.3f}, {svm_f1_ci[1]:.3f})")
print(f"SVM AUC-ROC: Mean={svm_roc_auc_mean:.3f}, CI=({svm_roc_auc_ci[0]:.3f}, {svm_roc_auc_ci[1]:.3f})")

# Create a DataFrame to store the performance metrics and confidence intervals
performance_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
    'Mean': [svm_accuracy_mean, svm_precision_mean, svm_recall_mean, svm_f1_mean, svm_roc_auc_mean],
    'Lower CI': [svm_accuracy_ci[0], svm_precision_ci[0], svm_recall_ci[0], svm_f1_ci[0], svm_roc_auc_ci[0]],
    'Upper CI': [svm_accuracy_ci[1], svm_precision_ci[1], svm_recall_ci[1], svm_f1_ci[1], svm_roc_auc_ci[1]]
})

plt.figure(figsize=(10, 6))
plt.title('Performance Metrics Boxplots')
sns.boxplot(data=[svm_accuracy_list, svm_precision_list, svm_recall_list, svm_f1_list, svm_roc_auc_list],
            orient='v',
            showfliers=True)
plt.xticks(ticks=range(5), labels=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'])
plt.xlabel('Performance Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


