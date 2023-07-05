import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ccase\Desktop\Dissertation\Datasets\perplexity_burstiness_scores.csv")

# # Plotting perplexity vs. label
# plt.figure(figsize=(10, 5))
# sns.histplot(data=df, x='perplexity', hue='AI Generated', bins=30)
# plt.title('Perplexity vs. AI Generated')
# plt.xlabel('Perplexity')
# plt.ylabel('Count')
# plt.legend(title='AI Generated')
# plt.show()
#
# # Plotting burstiness vs. label
# plt.figure(figsize=(10, 5))
# sns.histplot(data=df, x='burstiness', hue='AI Generated', bins=30)
# plt.title('Burstiness vs. AI Generated')
# plt.xlabel('Burstiness')
# plt.ylabel('Count')
# plt.legend(title='AI Generated')
# plt.show()
#
# # Plotting perplexity and burstiness together vs. label
# plt.figure(figsize=(10, 5))
# sns.histplot(data=df, x='perplexity', hue='AI Generated', bins=30, label='Perplexity')
# sns.histplot(data=df, x='burstiness', hue='AI Generated', bins=30, label='Burstiness')
# plt.title('Perplexity and Burstiness vs. AI Generated')
# plt.xlabel('Value')
# plt.ylabel('Count')
# plt.legend(title='Feature')
# plt.show()

# # Box plot for perplexity vs. label
# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df, x='AI Generated', y='perplexity')
# plt.title('Perplexity vs. AI Generated')
# plt.xlabel('AI Generated')
# plt.ylabel('Perplexity')
# plt.show()
#
# # Box plot for burstiness vs. label
# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df, x='AI Generated', y='burstiness')
# plt.title('Burstiness vs. AI Generated')
# plt.xlabel('AI Generated')
# plt.ylabel('Burstiness')
# plt.show()

# # Scatter plot for perplexity vs. burstiness, with separate plots for each label
# plt.figure(figsize=(10, 5))
#
# # Scatter plot for human-written (label 0)
# plt.subplot(1, 2, 1)
# sns.scatterplot(data=df[df['AI Generated'] == 0], x='perplexity', y='burstiness')
# plt.title('Perplexity vs. Burstiness (Human-Written)')
# plt.xlabel('Perplexity')
# plt.ylabel('Burstiness')
#
# # Scatter plot for AI-generated (label 1)
# plt.subplot(1, 2, 2)
# sns.scatterplot(data=df[df['AI Generated'] == 1], x='perplexity', y='burstiness')
# plt.title('Perplexity vs. Burstiness (AI-Generated)')
# plt.xlabel('Perplexity')
# plt.ylabel('Burstiness')
#
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# sns.lineplot(data=df, x='AI Generated', y='perplexity')
# plt.title('Perplexity Vs AI Generated')
# plt.xlabel('AI Generated')
# plt.ylabel('Perplexity')
# plt.show()

# plt.figure(figsize=(10, 5))
# sns.pointplot(data=df, x='AI Generated', y='perplexity')
# plt.title('Trend across Categories')
# plt.xlabel('AI Generated')
# plt.ylabel('perplexity')
# plt.show()

# plt.figure(figsize=(10, 5))
# sns.regplot(data=df, x='AI Generated', y='perplexity')
# plt.title('Relationship between AI Generated and perplexity')
# plt.xlabel('AI Generated')
# plt.ylabel('perplexity')
# plt.show()