# NLP Document Clustering - General Assignment (Task A)

This project is a Python implementation of an NLP pipeline that clusters a corpus of documents into a specified number of classes based on Cosine Similarity. It features an interactive, 3D animated visualization of the clustering process.

## Features
* **NLTK Integration:** Utilizes the Reuters news corpus as the dataset.
* **TF-IDF Vectorization:** Converts raw document text into meaningful numerical vectors.
* **Dimensionality Reduction:** Uses PCA to map complex high-dimensional text data into a 3D spatial environment.
* **Custom K-Means:** Implements clustering utilizing Cosine Similarity, strictly adhering to assignment parameters.
* **Interactive 3D Visualization:** An animated "neural-network" style graph where documents physically migrate to their cluster hubs.

## Prerequisites
You will need Python installed on your system along with the following libraries:
`pip install nltk scikit-learn matplotlib numpy`

*Note: On its first run, the script will automatically download the required NLTK Reuters data (~15MB) if you do not already have it.*

## How to Run
1. Open your terminal or command prompt.
2. Navigate to the directory containing the script.
3. Run the script using Python:
   `python clustering_animation.py`

## Interactive Interface
You do not need to edit the code to change the parameters. Upon running the script, the command-line interface will prompt you to define:

1. **Number of articles to process:** Controls the size of the corpus. (Recommended: 100 - 500. Higher numbers take longer to process and render).
2. **Number of clusters (k):** The amount of classes the algorithm will attempt to group the documents into. Colors are generated dynamically based on this number.

Just press `Enter` on your keyboard without typing a number to use the default values.

## Interacting with the Graph
Once the Matplotlib window opens and the animation begins:
* **Rotate Camera:** Left-click and hold anywhere in the graph, then drag your mouse to view the 3D space from any angle.
* **Zoom:** Use your mouse scroll wheel, or use the magnifying glass tool at the bottom of the window to zoom into specific clusters.