import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os

def cluster_near_duplicates(input_path, output_path, similarity_threshold=0.90):
    print("1. Loading stylometry dataset...")
    df = pd.read_csv(input_path)
    texts = df['text'].fillna("")
    n_reviews = len(df)
    
    print("2. Converting to TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    print("3. Finding nearest neighbors (Cosine Similarity)...")
    # k=10 is usually enough to link components of a cluster
    k_neighbors = min(10, n_reviews)
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', n_jobs=-1)
    nn.fit(X)
    
    distances, indices = nn.kneighbors(X)
    
    print(f"4. Building graph (connecting reviews with > {similarity_threshold*100}% similarity)...")
    # cosine distance = 1 - cosine similarity
    distance_threshold = 1.0 - similarity_threshold
    
    rows = []
    cols = []
    for i in range(n_reviews):
        for j_idx, j in enumerate(indices[i]):
            # distance of 0 is self, < threshold means very similar
            if distances[i, j_idx] < distance_threshold:
                rows.append(i)
                cols.append(j)
                
    # Create adjacency matrix
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_reviews, n_reviews))
    
    print("5. Extracting connected components (Template Clusters)...")
    n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    
    df['group_id'] = labels
    
    print(f"\n--- CLUSTERING RESULTS ---")
    print(f"Total Reviews: {n_reviews}")
    print(f"Total Unique Clusters Found: {n_components}")
    print(f"Number of duplicate templates identified: {n_reviews - n_components}")
    
    # Let's see some stats on cluster sizes
    cluster_sizes = pd.Series(labels).value_counts()
    print(f"Largest cluster size: {cluster_sizes.max()}")
    print(f"Number of clusters with >1 review: {(cluster_sizes > 1).sum()}")
    print("--------------------------\n")
    
    print(f"6. Saving grouped dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    input_file = os.path.join("data", "processed", "reviews_with_stylometry.csv")
    output_file = os.path.join("data", "processed", "reviews_with_groups.csv")
    cluster_near_duplicates(input_file, output_file, similarity_threshold=0.90)
