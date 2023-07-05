import pandas as pd


dataset = pd.read_csv(r'C:\Users\ccase\Desktop\Dissertation\Datasets\preprocessed_dataset_noTransformation.csv')

# print(dataset.head())


# Define a function to concatenate tokens in a list into a string
def concatenate_tokens(token_list):
    return ' '.join(token_list)


# dataset['text_to_analyse'] = dataset['text_to_analyse'].apply(concatenate_tokens)
dataset['text_to_analyse'] = [concatenate_tokens(record) for record in dataset['text_to_analyse']]
print(dataset.head())

# dataset = dataset.sample(frac=1).reset_index(drop=True)

# dataset.to_csv('preprocessed_detokenized_dataset_no_transformation.csv', index=False)

# tokens = ['herbert', 'george', 'webb', 'july', 'october', 'australian', 'mathematician', 'make', 'significant', 'contribution', 'abstract', 'algebra', 'topology', 'also', 'know', 'work', 'mathematical', 'physic', 'include', 'development', 'theory', 'black', 'hole', 'webb', 'bear', 'melbourne', 'july', 'son', 'herbert', 'webb', 'civil', 'engineer', 'attend', 'scotch', 'college', 'study', 'university', 'melbourne', 'receive', 'beginning', 'work', 'research', 'fellow', 'university', 'cambridge', 'receive', 'supervision', 'hardy', 'return', 'australia', 'webb', 'become', 'lecturer', 'university', 'melbourne', 'professor', 'serve', 'chairman', 'mathematics', 'department', 'melbourne', 'president', 'australian', 'mathematical', 'society', 'webb', 'die', 'october', 'melbourne', 'long', 'illness', 'memorial', 'service', 'hold', 'st', 'paul', 'cathedral', 'december', 'bury', 'springvale', 'cemetery']
#
# print(concatenate_tokens(tokens))