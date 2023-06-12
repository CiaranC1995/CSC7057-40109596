import csv
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_SpaceStory.txt') as f:
    text_to_analyse_1 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_AgileSoftwareDevelopment.txt') as f:
    text_to_analyse_2 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_CiaranSoftwareEngineer.txt') as f:
    text_to_analyse_3 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_ChatGPToutput.txt') as f:
    text_to_analyse_4 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_BraveNewWorldEssay.txt') as f:
    text_to_analyse_5 = f.read()

unprocessedTexts = [text_to_analyse, text_to_analyse_1, text_to_analyse_2, text_to_analyse_3, text_to_analyse_4,
                    text_to_analyse_5]

processed_texts = [pp.TextPreprocessor.remove_new_line_chars(text) for text in unprocessedTexts]

data_to_write = [[processed_texts[0], 0], [processed_texts[1], 1], [processed_texts[2], 1], [processed_texts[3], 1],
                 [processed_texts[4], 1], [processed_texts[5], 1]]


def create_and_add_to_csv_file(input_data):
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([['Text', 'AI-Generated']])
        writer.writerows(input_data)
    print("Data added successfully.")


create_and_add_to_csv_file(data_to_write)
