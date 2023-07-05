import ast
import datetime
import pandas as pd
from ydata_profiling import ProfileReport

csv = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\preprocessed_dataset_noTransformation.csv")

csv['text_to_analyse'] = [ast.literal_eval(string) for string in csv['text_to_analyse']]


def list_to_string(lst):
    return ' '.join(lst)


csv['text_to_analyse'] = csv['text_to_analyse'].apply(list_to_string)

csv = csv.sample(frac=1).reset_index(drop=True)

csv.to_csv('detokenized_preprocessed_dataset_noTransformation.csv', index=False)

# profile = ProfileReport(csv)
# profile.to_file(output_file="data_report_no_transformation.html")

print("Program finished executing at:", datetime.datetime.now())
