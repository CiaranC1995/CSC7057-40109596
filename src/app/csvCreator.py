import csv
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    text_to_analyse = f.read()

# with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_DanceWithDragons.txt') as f:
#     text_to_analyse1 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_DaVinciCode.txt') as f:
    text_to_analyse2 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_ClashOfKings.txt') as f:
    text_to_analyse3 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_BagOfBones.txt') as f:
    text_to_analyse4 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_SpaceStory.txt') as f:
    text_to_analyse5 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_AgileSoftwareDevelopment.txt') as f:
    text_to_analyse6 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_CiaranSoftwareEngineer.txt') as f:
    text_to_analyse7 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_ChatGPToutput.txt') as f:
    text_to_analyse8 = f.read()

with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_BraveNewWorldEssay.txt') as f:
    text_to_analyse9 = f.read()

unprocessedTexts = [text_to_analyse, text_to_analyse2, text_to_analyse3, text_to_analyse4,
                    text_to_analyse5, text_to_analyse6, text_to_analyse7, text_to_analyse8, text_to_analyse9]

# processed_texts = [pp.TextPreprocessor.remove_new_line_chars(text) for text in unprocessedTexts]

data_to_write = [[unprocessedTexts[0], 0], [unprocessedTexts[1], 0], [unprocessedTexts[2], 0], [unprocessedTexts[3], 0],
                 [unprocessedTexts[4], 1], [unprocessedTexts[5], 1], [unprocessedTexts[6], 1], [unprocessedTexts[7], 1],
                 [unprocessedTexts[8], 1]]


def create_and_add_to_csv_file(input_data):
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Text', 'AI-Generated'])
        for i, row in enumerate(input_data, start=1):
            writer.writerow([i, row[0], row[1]])
    print("Operation Successful.")


create_and_add_to_csv_file(data_to_write)
