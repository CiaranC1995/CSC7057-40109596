import datetime
import pickle
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import seaborn as sns

start_time = time.time()


# Function to evaluate SVM model and calculate performance metrics
def evaluate_svm_model(X_train, X_test, y_train, y_test):
    svm_model = LinearSVC(loss='squared_hinge', max_iter=10000, dual=False)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    pred_probabilities = svm_model._predict_proba_lr(X_test)

    # Measured performance metrics for predictions
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_recall = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)
    svm_roc_auc = roc_auc_score(y_test, y_pred_svm)

    # Sample indices for true AI Generated and true Human Written records in the validation set
    ai_indices = [i for i, val in enumerate(y_test) if val == 1]
    human_indices = [i for i, val in enumerate(y_test) if val == 0]

    # Model prediction probabilities for true AI Generated and true Human Written records in the validation set
    ai_pred_probabilities = pred_probabilities[ai_indices, 1]
    human_pred_probabilities = pred_probabilities[human_indices, 0]

    return svm_accuracy, svm_precision, svm_recall, svm_f1, svm_roc_auc, ai_pred_probabilities, human_pred_probabilities


# Training Dataset (80% of original Dataset, 20% kept aside for final model evaluation)
dataset = pd.read_csv(
    r"C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro\80percent_of_original_dataset.csv")

vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

# Extracting columns from Training Dataset
final_text = dataset['text_to_analyse'].to_list()
final_text_vectorized = vectorizer.transform(final_text)
final_ppl = dataset['perplexity'].to_list()
final_burstiness = dataset['burstiness'].to_list()
final_labels = dataset['AI Generated'].to_list()

# Due to large memory and computational requirements, it was necessary to split bootstrapping training validation into
# batches and take averages of the performance metrics.
batch_size = 40000
# Bootstrapping for 1000 iterations
num_iterations = 1000

svm_accuracy_list = []
svm_precision_list = []
svm_recall_list = []
svm_f1_list = []
svm_roc_auc_list = []
svm_human_prob = []
svm_ai_prob = []

print(f'Beginning Iterations @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# Loop for Bootstrapping
for iteration_count, i in enumerate(range(num_iterations)):

    # Different 80% / 20% split of the training dataset for each bootstrap

    X_train_text, X_test_text, X_train_perplexity, X_test_perplexity, X_train_burstiness, X_test_burstiness, y_train, y_test = train_test_split(
        final_text_vectorized, final_ppl, final_burstiness, final_labels, test_size=0.2)

    # Batch size for horizontally stacking the features to make the final model input arrays
    hstack_batch_size = 10000

    # Create empty arrays to store the final data
    final_x_train_list = []
    final_x_test_list = []

    # Iterate over batches
    for i in range(0, len(X_train_perplexity), hstack_batch_size):
        X_train_batch = X_train_text[i:i + hstack_batch_size].toarray()
        X_test_batch = X_test_text[i:i + hstack_batch_size].toarray()

        # Creating the arrays that will eventually be model inputs
        X_train_batch = np.hstack((X_train_batch, np.array(X_train_perplexity[i:i + hstack_batch_size]).reshape(-1, 1),
                                   np.array(X_train_burstiness[i:i + hstack_batch_size]).reshape(-1, 1)))
        X_test_batch = np.hstack((X_test_batch, np.array(X_test_perplexity[i:i + hstack_batch_size]).reshape(-1, 1),
                                  np.array(X_test_burstiness[i:i + hstack_batch_size]).reshape(-1, 1)))

        final_x_train_list.append(X_train_batch)
        final_x_test_list.append(X_test_batch)

    # Combine the batches into final arrays
    final_x_train = np.vstack(final_x_train_list)
    final_x_test = np.vstack(final_x_test_list)

    print(f'Stacking Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

    sample_indices = np.random.choice(range(len(X_train_perplexity)), size=len(X_train_perplexity), replace=False)
    num_batches = len(sample_indices) // batch_size
    batches = np.array_split(sample_indices, num_batches)

    # Initialize lists to store metrics for each batch
    batch_accuracy = []
    batch_precision = []
    batch_recall = []
    batch_f1 = []
    batch_auc = []
    batch_human_prob = []
    batch_ai_prob = []

    for batch_count, batch_indices in enumerate(batches):
        # Generate bootstrap samples for the current batch
        X_train_bootstrap = final_x_train[batch_indices]
        y_train_bootstrap = np.array(y_train)[batch_indices]

        # Evaluate SVM model on the current batch
        acc, prec, rec, f1, auc, ai_prob, human_prob = evaluate_svm_model(X_train_bootstrap, final_x_test,
                                                                          y_train_bootstrap, y_test)

        print(f'Batch Model Evaluation Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

        # Append batch metrics to corresponding lists
        batch_accuracy.append(acc)
        batch_precision.append(prec)
        batch_recall.append(rec)
        batch_f1.append(f1)
        batch_auc.append(auc)
        svm_human_prob.extend(human_prob)
        svm_ai_prob.extend(ai_prob)

        # Print the count during each batch iteration
        print(
            f'Iteration {iteration_count + 1}/{num_iterations}, Batch {batch_count + 1}/{num_batches} Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')
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

ai_mean_prob = np.mean(svm_ai_prob)
ai_ci = np.percentile(svm_ai_prob, [2.5, 97.5])

human_mean_prob = np.mean(svm_human_prob)
human_ci = np.percentile(svm_human_prob, [2.5, 97.5])

print(f"AI Predicted Probability: Mean={ai_mean_prob:.3f}, CI=({ai_ci[0]:.3f}, {ai_ci[1]:.3f})")
print(f"Human Predicted Probability: Mean={human_mean_prob:.3f}, CI=({human_ci[0]:.3f}, {human_ci[1]:.3f}")

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

print("Time Elapsed: {:.2f}s".format(time.time() - start_time))
print("Program finished executing at:", datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))

# Plot the measured performance metrics on vertical box plots
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
