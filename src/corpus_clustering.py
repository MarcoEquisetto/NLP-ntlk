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
fileids = reuters.fileids()[:500]
docs = [reuters.raw(fileid) for fileid in fileids]

# 2. Vectorize the text using TF-IDF
print("Computing TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(docs).toarray()

# 3. Reduce dimensionality to 3D for spatial visualization
print("Reducing dimensionality to 3D...")
pca = PCA(n_components=3)
target_X = pca.fit_transform(X) 

# 4. Pre-compute Custom K-Means using Cosine Similarity
def cosine_distance(points, centers):
    norm_points = np.linalg.norm(points, axis=1, keepdims=True)
    norm_centers = np.linalg.norm(centers, axis=1)
    norm_points[norm_points == 0] = 1e-10
    norm_centers[norm_centers == 0] = 1e-10
    dot_product = np.dot(points, centers.T)
    cos_similarity = dot_product / (norm_points * norm_centers)
    return 1 - cos_similarity

print("Pre-computing clusters...")
k = 3
np.random.seed(42)
initial_indices = np.random.choice(target_X.shape[0], k, replace=False)
centroids = target_X[initial_indices]
labels = np.zeros(target_X.shape[0])

# Run K-means to convergence
for _ in range(50): 
    distances = cosine_distance(target_X, centroids)
    labels = np.argmin(distances, axis=1)
    new_centroids = np.array([target_X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

# 5. Set up initial positions
min_x, max_x = np.min(target_X[:, 0]), np.max(target_X[:, 0])
min_y, max_y = np.min(target_X[:, 1]), np.max(target_X[:, 1])
max_z = np.max(target_X[:, 2])
min_z = np.min(target_X[:, 2]) - 0.2  

current_X = np.column_stack((
    np.random.uniform(min_x, max_x, size=target_X.shape[0]), 
    np.random.uniform(min_y, max_y, size=target_X.shape[0]),
    np.full(target_X.shape[0], min_z - 0.5) 
))

# 6. Plotting Setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Reuters Clustering (Cosine Similarity)\nClick and Drag to Rotate Camera!", pad=20)

# --- NEW: Spatial Reference Styling ---
# Make the background panes transparent so it doesn't look boxed in
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Make the pane edges invisible or very faint
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Style the grid lines to be subtle
ax.grid(True, linestyle=':', alpha=0.3, color='gray')

# Draw explicit reference lines (X, Y, Z axes) passing through the origin (0,0,0)
ax.plot([min_x, max_x], [0, 0], [0, 0], color='black', linestyle='--', alpha=0.5, lw=1.5, zorder=0)
ax.plot([0, 0], [min_y, max_y], [0, 0], color='black', linestyle='--', alpha=0.5, lw=1.5, zorder=0)
ax.plot([0, 0], [0, 0], [min_z, max_z], color='black', linestyle='--', alpha=0.5, lw=1.5, zorder=0)
# --------------------------------------

ax.set_xlim(min_x - 0.1, max_x + 0.1)
ax.set_ylim(min_y - 0.1, max_y + 0.1)
ax.set_zlim(min_z - 0.6, max_z + 0.1)

# Axis labels for orientation
ax.set_xlabel('PCA X')
ax.set_ylabel('PCA Y')
ax.set_zlabel('PCA Z')

colors = plt.cm.rainbow(np.linspace(0, 1, k))
point_colors = [colors[label] for label in labels]

centroid_scatter = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=colors, s=250, marker='*', zorder=4, edgecolors='black')

scatter = ax.scatter(current_X[:, 0], current_X[:, 1], current_X[:, 2], c=point_colors, s=20, zorder=3, edgecolors='w', linewidth=0.5)

lines = [ax.plot([], [], [], c=colors[labels[i]], lw=0.5, alpha=0.3, zorder=1)[0] 
         for i in range(target_X.shape[0])]

def update(frame):
    global current_X
    
    smoothness = 0.1 
    current_X = current_X * (1 - smoothness) + target_X * smoothness
    
    scatter._offsets3d = (current_X[:, 0], current_X[:, 1], current_X[:, 2])
    
    for i in range(current_X.shape[0]):
        cluster_idx = labels[i]
        c_x, c_y, c_z = centroids[cluster_idx]
        p_x, p_y, p_z = current_X[i]
        
        lines[i].set_data([p_x, c_x], [p_y, c_y])
        lines[i].set_3d_properties([p_z, c_z])
        
    return [scatter] + lines

print("Starting animation...")
ani = animation.FuncAnimation(fig, update, frames=150, interval=20, blit=False)

plt.tight_layout()
plt.show()