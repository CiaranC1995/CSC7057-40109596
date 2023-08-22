import datetime
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

# Import and format test samples
test_dataset = pd.read_csv(
    r"C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro\20percent_of_original_dataset_FINAL_TEST.csv")

num_samples = 25

human_dataset = test_dataset[test_dataset["AI Generated"] == 0]
ai_dataset = test_dataset[test_dataset["AI Generated"] == 1]

human_sampled_records = human_dataset.sample(n=num_samples, random_state=42)
ai_sampled_records = ai_dataset.sample(n=num_samples, random_state=42)

combined_sampled_records = pd.concat([human_sampled_records, ai_sampled_records], ignore_index=True)

final_testing_dataset = combined_sampled_records.sample(frac=1, random_state=42)

# Prepare for model training
vectorizer_path = r'./Vectorizers/tfidfvectorizer.pickle'

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

final_text = final_testing_dataset['text_to_analyse'].to_list()
final_text_vectorized = vectorizer.transform(final_text)
final_ppl = final_testing_dataset['perplexity'].to_list()
final_burstiness = final_testing_dataset['burstiness'].to_list()
final_labels = final_testing_dataset['AI Generated'].to_list()

final_x_test = np.hstack((final_text_vectorized.toarray(), np.array(final_ppl).reshape(-1, 1),
                          np.array(final_burstiness).reshape(-1, 1)))

y_test = final_labels

# Import model and carry out predictions
model_path = r'./Models/LinearSVC_CV_80percentDataset.pickle'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

y_pred_svm = model.predict(final_x_test)
pred_probabilities = model._predict_proba_lr(final_x_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_roc_auc = roc_auc_score(y_test, y_pred_svm)

print(f"SVM Accuracy: {(svm_accuracy * 100):.3f}%")
print(f"SVM Precision: {svm_precision:.3f}")
print(f"SVM Recall: {svm_recall:.3f}")
print(f"SVM F1-score: {svm_f1:.3f}")
print(f"SVM AUC-ROC: {svm_roc_auc:.3f}")


# **********************************************************************************************************************
def evaluate_svm_model(X_train, X_test, y_train, y_test):
    svm_model = LinearSVC(loss='squared_hinge', max_iter=10000, dual=False)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    pred_probabilities = svm_model._predict_proba_lr(X_test)

    return y_pred_svm, pred_probabilities


training_dataset = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro"
                               r"\80percent_of_original_dataset.csv")

# Bootstrapping and predicting 1,000 times on random 90% of final training dataset
num_bootstraps = 5
num_samples = int(0.9 * len(training_dataset))

sampled_indices = final_testing_dataset.index.tolist()  # Indices of samples in the test set

# Prediction Indices
predicted_human_indices = [index for index, label in enumerate(y_pred_svm) if label == 0]
predicted_ai_indices = [index for index, label in enumerate(y_pred_svm) if label == 1]

sample_dictionary = {sample_index: {"prediction": [], "human_probabilities": [], "ai_probabilities": []} for sample_index in
                     sampled_indices}  # Initialize a dictionary

for iteration_count, i in enumerate(range(num_bootstraps)):
    # Randomly sample 90% of the dataset
    sampled_dataset = training_dataset.sample(n=num_samples)

    # text = sampled_dataset['text_to_analyse'].to_list()
    # text_vectorized = vectorizer.transform(text)
    # ppl = sampled_dataset['perplexity'].to_list()
    # burstiness = sampled_dataset['burstiness'].to_list()
    # labels = sampled_dataset['AI Generated'].to_list()

    text = sampled_dataset['text_to_analyse'].head(50000).to_list()
    text_vectorized = vectorizer.transform(text)
    ppl = sampled_dataset['perplexity'].head(50000).to_list()
    burstiness = sampled_dataset['burstiness'].head(50000).to_list()
    labels = sampled_dataset['AI Generated'].head(50000).to_list()

    bootstrap_x_train = np.hstack((text_vectorized.toarray(), np.array(ppl).reshape(-1, 1),
                                   np.array(burstiness).reshape(-1, 1)))

    bootstrap_y_train = labels

    preds, preds_probs = evaluate_svm_model(bootstrap_x_train, final_x_test, bootstrap_y_train, y_test)

    # Collect prediction and probabilities for each sample
    for sample_index, (prediction, probs) in zip(sampled_indices, zip(preds, preds_probs)):
        if prediction == 0:
            sample_dictionary[sample_index]["prediction"].append(prediction)
            sample_dictionary[sample_index]["human_probabilities"].append((probs[0]))
        else:
            sample_dictionary[sample_index]["prediction"].append(prediction)
            sample_dictionary[sample_index]["ai_probabilities"].append((probs[1]))

    print(f'Iteration {iteration_count+1} / {num_bootstraps} Complete @ {datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")}')

# Print the sample probabilities
# for sample_index, info in sorted(sample_dictionary.items()):
#     print(f"Sample {sample_index + 1} Prediction: {info['prediction']}")
#     print(f"Sample {sample_index + 1} Human Probabilities: {info['human_probabilities']}")
#     print(f"Sample {sample_index + 1} AI Probabilities: {info['ai_probabilities']}")

# Plotting box plots

human_prediction_probs = [sample_dictionary[index]["human_probabilities"] for index in sample_dictionary]
ai_prediction_probs = [sample_dictionary[index]["ai_probabilities"] for index in sample_dictionary]

# print(human_prediction_probs)
# print(len(human_prediction_probs))
#
# print(ai_prediction_probs)
# print(len(ai_prediction_probs))

initial_preds = y_pred_svm
initial_pred_probabilities = pred_probabilities
initial_unique_probabilities = []

for pred, prob in zip(initial_preds, initial_pred_probabilities):
    if pred == 0:
        initial_unique_probabilities.append(prob[0])
    elif pred == 1:
        initial_unique_probabilities.append(prob[1])

initial_human_pred_probs = [initial_unique_probabilities[index] for index in predicted_human_indices]
initial_ai_pred_probs = [initial_unique_probabilities[index] for index in predicted_ai_indices]

# Plotting box plots
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=human_prediction_probs)
plt.xticks()
plt.xlabel('Sample Index')
plt.ylabel('Prediction Probability')
plt.title('Box Plots of Human Written Prediction Probabilities for All Samples')

# Plot initial_unique_probabilities as circles on top of the boxplots
# for index, prob in enumerate(initial_human_pred_probs):
#     if index in predicted_human_indices:
#         plt.scatter(index, prob, color='red', marker='o', s=30, zorder=5)

xtick_labels = [int(label.get_text()) + 1 for label in ax.get_xticklabels()]
ax.set_xticklabels(xtick_labels)

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=ai_prediction_probs)
plt.xticks()
plt.xlabel('Sample Index')
plt.ylabel('Prediction Probability')
plt.title('Box Plots of AI Generated Prediction Probabilities for All Samples')

# Plot initial_unique_probabilities as circles on top of the boxplots
# for index, prob in enumerate(initial_ai_pred_probs):
#     if index in predicted_ai_indices:
#         plt.scatter(index, prob, color='red', marker='o', s=30, zorder=5)

xtick_labels = [int(label.get_text()) + 1 for label in ax.get_xticklabels()]
ax.set_xticklabels(xtick_labels)

plt.tight_layout()
plt.show()



