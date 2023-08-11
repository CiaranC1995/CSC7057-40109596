import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

datasets = pd.read_csv(r'C:\Users\ccase\Desktop\Dissertation\Datasets\GPT-wiki-intro\vectorized_dataframe.csv')

documents = datasets['original_text']

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)

# TruncatedSVD for dimensionality reduction (analogous to PCA for sparse data)
n_components = 3  # Number of components for dimensionality reduction
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
feature_names = svd.get_feature_names_out()
print(feature_names)

# t-SNE for further dimensionality reduction and visualization
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_svd)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_svd[:, 0], X_svd[:, 1])
plt.title("TruncatedSVD Visualization")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.show()
