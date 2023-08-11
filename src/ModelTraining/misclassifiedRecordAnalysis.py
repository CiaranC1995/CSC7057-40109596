import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    r"C:\Users\ccase\Desktop\Dissertation\Documents\Data Visualisation\NEW "
    r"PERFORMANCE\80_train_same_80_val_no_boot\Stats\Misclassified\cross_validated_misclassified_records.csv")

title = 'All Misclassified Records'
# title = 'Records Misclassified as AI Generated'
# title = 'Records Misclassified as Human Written'

variable = 'Prediction Probability'

no_filter_dataframe = dataset
ai_pred_filtered_dataframe = dataset[dataset['predicted_label'] == 1]
human_pred_filtered_dataframe = dataset[dataset['predicted_label'] == 0]

column_data_unfiltered = no_filter_dataframe['prediction_probability']

print('Total Number of Records Misclassified : ', len(dataset))
# print(f'Number of {title} : {len(ai_pred_filtered_dataframe)}')
# print(f'Number of {title} : {len(human_pred_filtered_dataframe)}')
print()

mean = column_data_unfiltered.mean()
maximum = max(column_data_unfiltered)
minimum = min(column_data_unfiltered)
median = column_data_unfiltered.median()
std_deviation = column_data_unfiltered.std()
variance = column_data_unfiltered.var()
q1 = column_data_unfiltered.quantile(0.25)
q2 = column_data_unfiltered.quantile(0.50)
q3 = column_data_unfiltered.quantile(0.75)
iqr = q3 - q1

print(f'Mean {variable} of {title} : {mean}')
print(f'Median {variable} of {title} : {median}')
print()
print(f'Maximum {variable} of {title} : {maximum}')
print(f'Minimum {variable} of {title} : {minimum}')
print(f'Range of {variable} of {title} : {maximum - minimum}')
print()
print(f'Variance of {variable} of {title} : {variance}')
print(f'Standard Deviation of {variable} of {title} : {std_deviation}')
print()
print(f"Quartile 1 (Q1) : {q1}")
print(f"Quartile 2 (Q2) or Median : {q2}")
print(f"Quartile 3 (Q3) : {q3}")
print(f"Interquartile Range (IQR) : {iqr}")
print()

# # Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(column_data_unfiltered, bins=20, color='skyblue', edgecolor='black')
plt.xlabel(variable)
plt.ylabel('Frequency')
plt.title(f'Histogram of {title}')
plt.grid(True)
plt.show()

# Plot box plot
plt.figure(figsize=(8, 6))
plt.boxplot(column_data_unfiltered, vert=False)
plt.xlabel(variable)
plt.title(f'Box Plot of {title}')
plt.grid(True)
plt.show()

# Plot KDE (Kernel Density Estimation) plot
plt.figure(figsize=(8, 6))
column_data_unfiltered.plot(kind='kde', color='blue')
plt.xlabel(variable)
plt.ylabel('Density')
plt.title(f'KDE Plot of {title}')
plt.grid(True)
plt.show()

# Summary statistics
# summary_stats = column_data.describe()
# print(f'Summary Statistics for {title} :')
# print(summary_stats)
