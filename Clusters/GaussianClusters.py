from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from user_embedding import final_df

embeddings_df = final_df.drop(columns=['id', 'age','tags'])  # Get the embeddings columns

# Dimensionality Reduction (PCA) for better clustering
pca = PCA(n_components=10)  # Reduce dimensions to 10
reduced_embeddings = pca.fit_transform(embeddings_df)

# Perform GMM clustering (soft clustering)
n_clusters = 5  # figure out way to make it dynamic according to how many users there are
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
final_df['cluster'] = gmm.fit_predict(reduced_embeddings)

# Check the clustering result
print(final_df[['id', 'age', 'tags', 'cluster']])

import matplotlib.pyplot as plt
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=final_df['cluster'], cmap='viridis')
plt.colorbar()
plt.show()
