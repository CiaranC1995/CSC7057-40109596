import csv
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()


def process_quotation_marks(input_text):
    return input_text.replace('"', '""')


processed_text = pp.TextPreprocessor.remove_new_line_chars(text_to_analyse)
csv_text = process_quotation_marks(processed_text)

data_to_write = [[csv_text, 1]]
print(data_to_write)
print(data_to_write[0][1])


def create_csv_file(input_data):
    # Create the CSV file
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([['ID', 'Text', 'AI-Generated']])
        text_id = 1
        for text in range(len(input_data)):
            writer.writerow([[text_id, input_data[text_id - 1][1], input_data[text_id - 1][0]]])
            text_id += 1
    print("CSV file 'test.csv' created successfully.")

# Call the function
create_csv_file(data_to_write)
