import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    r"C:\Users\ccase\Desktop\Dissertation\Documents\Data Visualisation\NEW "
    r"PERFORMANCE\80_train_same_80_val_no_boot\Stats\Misclassified\cross_validated_misclassified_records.csv")

# title = 'All Misclassified Records'
title = 'Records Misclassified as AI Generated'
# title = 'Records Misclassified as Human Written'

variable_title = 'Perplexity'
csv_variable_name = 'perplexity'

no_filter_dataframe = dataset
ai_pred_filtered_dataframe = dataset[dataset['predicted_label'] == 1]
human_pred_filtered_dataframe = dataset[dataset['predicted_label'] == 0]

column_data_unfiltered = ai_pred_filtered_dataframe[csv_variable_name]

# print('Total Number of Records Misclassified : ', len(dataset))
print(f'Number of {title} : {len(ai_pred_filtered_dataframe)}')
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

print(f'Mean {variable_title} of {title} : {mean}')
print(f'Median {variable_title} of {title} : {median}')
print()
print(f'Maximum {variable_title} of {title} : {maximum}')
print(f'Minimum {variable_title} of {title} : {minimum}')
print(f'Range of {variable_title} of {title} : {maximum - minimum}')
print()
print(f'Variance of {variable_title} of {title} : {variance}')
print(f'Standard Deviation of {variable_title} of {title} : {std_deviation}')
print()
print(f"Quartile 1 (Q1) : {q1}")
print(f"Quartile 2 (Q2) or Median : {q2}")
print(f"Quartile 3 (Q3) : {q3}")
print(f"Interquartile Range (IQR) : {iqr}")
print()

normal_lower_data_range = q1 - (1.5 * mean)
normal_upper_data_range = q3 + (1.5 * mean)

print(f'Normal Upper Data Range (q3 + (1.5 * mean)) : {normal_upper_data_range}')
print(f'Normal Lower Data Range (q1 - (1.5 * mean)) : {normal_lower_data_range}')
print()

dataframe_no_outliers = ai_pred_filtered_dataframe[(ai_pred_filtered_dataframe[csv_variable_name] >= normal_lower_data_range) & (ai_pred_filtered_dataframe[csv_variable_name] <= normal_upper_data_range)]

num_points_above_upper_limit = len(ai_pred_filtered_dataframe[ai_pred_filtered_dataframe[csv_variable_name] > normal_upper_data_range])
num_points_below_lower_limit = len(ai_pred_filtered_dataframe[ai_pred_filtered_dataframe[csv_variable_name] < normal_lower_data_range])

print(f'Number of Outliers above Upper Data Range : {num_points_above_upper_limit}')
print(f'Number of Outliers below Lower Data Range : {num_points_below_lower_limit}')

# Plot histogram
# plt.figure(figsize=(8, 6))
# plt.hist(column_data_unfiltered, bins=20, color='skyblue', edgecolor='black')
# plt.xlabel(variable_title)
# plt.ylabel('Frequency')
# plt.title(f'Histogram of {title}')
# plt.grid(True)
# plt.show()
#
# # Plot box plot
# plt.figure(figsize=(8, 6))
# plt.boxplot(column_data_unfiltered, vert=False)
# plt.xlabel(variable_title)
# plt.title(f'Box Plot of {title}')
# plt.grid(True)
# plt.show()
#
# # Plot KDE (Kernel Density Estimation) plot
# plt.figure(figsize=(8, 6))
# column_data_unfiltered.plot(kind='kde', color='blue')
# plt.xlabel(variable_title)
# plt.ylabel('Density')
# plt.title(f'KDE Plot of {title}')
# plt.grid(True)
# plt.show()

print()
print('*************************************************************************************************************************************************************')
print()
print(f'Stats For Data with Outliers Removed')
print()
column_data_no_outliers = dataframe_no_outliers[csv_variable_name]

mean_no_outlier = column_data_no_outliers.mean()
maximum_no_outlier = max(column_data_no_outliers)
minimum_no_outlier = min(column_data_no_outliers)
median_no_outlier = column_data_no_outliers.median()
std_deviation_no_outlier = column_data_no_outliers.std()
variance_no_outlier = column_data_no_outliers.var()
q1_no_outlier = column_data_no_outliers.quantile(0.25)
q2_no_outlier = column_data_no_outliers.quantile(0.50)
q3_no_outlier = column_data_no_outliers.quantile(0.75)
iqr_no_outlier = q3 - q1

print(f'Mean {variable_title} of {title} : {mean_no_outlier}')
print(f'Median {variable_title} of {title} : {median_no_outlier}')
print()
print(f'Maximum {variable_title} of {title} : {maximum_no_outlier}')
print(f'Minimum {variable_title} of {title} : {minimum_no_outlier}')
print(f'Range of {variable_title} of {title} : {maximum_no_outlier - minimum_no_outlier}')
print()
print(f'Variance of {variable_title} of {title} : {variance_no_outlier}')
print(f'Standard Deviation of {variable_title} of {title} : {std_deviation_no_outlier}')
print()
print(f"Quartile 1 (Q1) : {q1_no_outlier}")
print(f"Quartile 2 (Q2) or Median : {q2_no_outlier}")
print(f"Quartile 3 (Q3) : {q3_no_outlier}")
print(f"Interquartile Range (IQR) : {iqr_no_outlier}")
print()

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(column_data_no_outliers, bins=20, color='skyblue', edgecolor='black')
plt.xlabel(variable_title)
plt.ylabel('Frequency')
plt.title(f'Histogram of {title} with Outliers Removed')
plt.grid(True)
plt.show()

# Plot box plot
plt.figure(figsize=(8, 6))
plt.boxplot(column_data_no_outliers, vert=False)
plt.xlabel(variable_title)
plt.title(f'Box Plot of {title} with Outliers Removed')
plt.grid(True)
plt.show()

# Plot KDE (Kernel Density Estimation) plot
plt.figure(figsize=(8, 6))
column_data_no_outliers.plot(kind='kde', color='blue')
plt.xlabel(variable_title)
plt.ylabel('Density')
plt.title(f'KDE Plot of {title} with Outliers Removed')
plt.grid(True)
plt.show()

# Summary statistics
# summary_stats = column_data.describe()
# print(f'Summary Statistics for {title} :')
# print(summary_stats)
