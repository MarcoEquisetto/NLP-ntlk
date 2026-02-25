import nltk
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1. Download necessary NLTK data
try:
    nltk.data.find('corpora/reuters.zip')
except LookupError:
    nltk.download('reuters')
    nltk.download('punkt')

print("Loading Reuters corpus subset...")
fileids = reuters.fileids()[:300]
docs = [reuters.raw(fileid) for fileid in fileids]

# 2. Vectorize the text using TF-IDF
print("Computing TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(docs).toarray()

# 3. Reduce dimensionality to 2D for visualization
print("Reducing dimensionality...")
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
# Note: We are no longer forcing these into a circle. They will keep their natural shape.

# 4. Custom K-Means using Cosine Similarity
k = 3
np.random.seed(42)
initial_indices = np.random.choice(X_2d.shape[0], k, replace=False)
centroids = X_2d[initial_indices]
labels = np.zeros(X_2d.shape[0])

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Reuters Document Clustering (Cosine Similarity)\nPoints Migrating to Centroids", pad=20)
ax.axis('off') 

colors = ['#FF5733', '#33C1FF', '#8D33FF']

scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c='gray', s=20, zorder=3, edgecolors='w', linewidth=0.5)
centroid_scatter = ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=150, marker='*', zorder=4)
lines = [ax.plot([], [], c='gray', lw=0.5, alpha=0.3, zorder=1)[0] for _ in range(X_2d.shape[0])]

def cosine_distance(points, centers):
    """Calculates cosine distance between N points and K centers."""
    norm_points = np.linalg.norm(points, axis=1, keepdims=True)
    norm_centers = np.linalg.norm(centers, axis=1)
    
    # Avoid division by zero
    norm_points[norm_points == 0] = 1e-10
    norm_centers[norm_centers == 0] = 1e-10
    
    dot_product = np.dot(points, centers.T)
    cos_similarity = dot_product / (norm_points * norm_centers)
    return 1 - cos_similarity # Distance is 1 - similarity

def update(frame):
    global centroids, labels
    
    # Step 1: Assign points based on Cosine Distance
    distances = cosine_distance(X_2d, centroids)
    labels = np.argmin(distances, axis=1)
    
    # Step 2: Update centroids
    new_centroids = np.array([X_2d[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
    
    # Interpolate for slower, smoother animation
    smoothness = 0.5 # Lowered to make the migration slower
    centroids = centroids * (1 - smoothness) + new_centroids * smoothness
    
    # Update visuals
    point_colors = [colors[label] for label in labels]
    scatter.set_color(point_colors)
    scatter.set_edgecolor('w')
    
    centroid_scatter.set_offsets(centroids)
    centroid_scatter.set_color(colors)
    
    for i in range(X_2d.shape[0]):
        cluster_idx = labels[i]
        c_x, c_y = centroids[cluster_idx]
        p_x, p_y = X_2d[i]
        lines[i].set_data([p_x, c_x], [p_y, c_y])
        lines[i].set_color(colors[cluster_idx])
        lines[i].set_alpha(0.4)
        
    return [scatter, centroid_scatter] + lines

print("Starting animation...")
# Increased interval from 50ms to 200ms to slow down the frame rate
ani = animation.FuncAnimation(fig, update, frames=150, interval=200, blit=True)

plt.tight_layout()
plt.show()