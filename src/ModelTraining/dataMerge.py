import os
import pandas as pd


def read_txt_file(file_path, encoding="utf-8"):
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
    return content


human_directory_path = r'C:\Users\ccase\Desktop\Dissertation\Datasets\chatgpt-generated-text-corpus\full_texts\human'
chatgpt_directory_path = r'C:\Users\ccase\Desktop\Dissertation\Datasets\chatgpt-generated-text-corpus\full_texts' \
                         r'\chatgpt'

# HUMAN TEXT

human_file_contents_list = []
human_filename_list = []

for human_filename in os.listdir(human_directory_path):
    if human_filename.endswith(".txt"):
        file_path = os.path.join(human_directory_path, human_filename)
        file_content = read_txt_file(file_path)
        human_file_contents_list.append(file_content)
        human_filename_list.append(human_filename)

df_human = pd.DataFrame()
df_human['Human_Text'] = human_file_contents_list

# CHATGPT TEXT

chatgpt_file_contents_list = []
chatgpt_filename_list = []

for chatgpt_filename in os.listdir(chatgpt_directory_path):
    if chatgpt_filename.endswith(".txt"):
        file_path = os.path.join(chatgpt_directory_path, chatgpt_filename)
        file_content = read_txt_file(file_path)
        chatgpt_file_contents_list.append(file_content)
        chatgpt_filename_list.append(chatgpt_filename)

df_mixed = pd.DataFrame()

df_mixed['Filename'] = human_filename_list + chatgpt_filename_list
df_mixed['text_to_analyse'] = human_file_contents_list + chatgpt_file_contents_list
# df_mixed["AI_Generated"] = [0 for i in range(len(df_human))] + [1 for i in range(len(df_human))]

df_mixed.to_csv("new_dataset_github.csv", index=False)
