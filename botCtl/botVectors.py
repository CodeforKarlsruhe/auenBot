import os
import json
import numpy as np

#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
#from ragInstrumentation import measure_execution_time

DEBUG = False

def load_vectors(filename):
    """
    Load and normalize vectors from a JSON file.
    Args:
        filename (str): Path to the JSON file containing float32 vectors.
    Returns:
        np.ndarray: A 2D numpy array of shape (N, args.dim) containing the normalized vectors.
        list: A list of intents corresponding to each vector.
    Raises:
        ValueError: If the file size is not divisible by the size of a single vector record.
    Notes:
        - The JSON file is expected to contain vectors of shape (N, args.dim) where each element is a float32.
        - Each vector is normalized to unit length.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    with open(filename, 'r') as f:
        raw = json.load(f)

    if raw.get("vectors") is None:
        raise ValueError("Invalid file format: 'vectors' key not found.")
    vectors = np.array(raw["vectors"], dtype=np.float32)
    if raw.get("intents") is None:
        raise ValueError("Invalid file format: 'intents' key not found.")
    intents = raw["intents"]

    print(f"Number of vectors: {len(vectors)}, shape: {vectors.shape}")

    # Normalize each vector to unit length.
    # Axis=1 means we take the norm across each row (each vector).
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors_normalized = vectors / norms
    
    return vectors_normalized, intents


#########################
def compute_cosine_similarity(query: np.ndarray, doc: np.ndarray) -> float:
    """
        Parameters:
        query (np.ndarray): The first vector.
        doc (np.ndarray): The second vector.

        Returns:
        float: The cosine similarity between the two vectors.
    """
    # np.dot is efficient for dot products
    # Use a small epsilon check to avoid dividing by zero
    denom = (np.linalg.norm(query) * np.linalg.norm(doc)) + 1e-9
    return np.dot(query, doc) / denom

def query_vectors(vectors, query, num_neighbors=5) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest neighbors to a query vector from a set of vectors.
    Parameters:
    vectors (numpy.ndarray): A 2D array where each row is a vector from the dataset.
    query (numpy.ndarray): A 1D array representing the query vector.
    num_neighbors (int, optional): The number of nearest neighbors to return. Default is 5.
    Returns:
    tuple: A tuple containing two elements:
        - indices (numpy.ndarray): The indices of the nearest neighbors in the dataset.
        - distances (numpy.ndarray): The distances of the nearest neighbors from the query vector.
    """
    # Normalize the query vector
    query_norm = np.linalg.norm(query) + 1e-9
    query_normalized = query / query_norm

    query = query_normalized.flatten()
    similarities = [compute_cosine_similarity(query, v) for v in vectors]
    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_sims = [similarities[i].astype(float) for i in sorted_indices]
    
    if num_neighbors > 0:
        return sorted_indices[:num_neighbors], sorted_sims[:num_neighbors]
    else:
        return sorted_indices, sorted_sims

