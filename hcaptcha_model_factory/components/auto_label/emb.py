import numpy as np


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors
    :param a: vector a
    :param b: vector b
    :returns: cosine similarity
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def l2_distance(a, b):
    """Calculate L2 distance between two vectors
    :param a: vector a
    :param b: vector b
    :returns: L2 distance
    """
    return np.linalg.norm(a - b)


def l1_distance(a, b):
    """Calculate L1 distance between two vectors
    :param a: vector a
    :param b: vector b
    :returns: L1 distance
    """
    return np.sum(np.abs(a - b))


def euclidean_distance(a, b):
    """Calculate Euclidean distance between two vectors
    :param a: vector a
    :param b: vector b
    :returns: Euclidean distance
    """
    return np.sqrt(np.sum(np.square(a - b)))


def cosine_distance(a, b):
    """Calculate cosine distance between two vectors
    :param a: vector a
    :param b: vector b
    :returns: cosine distance
    """
    return 1 - cosine_similarity(a, b)


def get_distance_function(distance):
    """Get distance function
    :param distance: distance function name
    :returns: distance function
    """
    if distance == "cosine":
        return cosine_distance
    elif distance == "euclidean":
        return euclidean_distance
    elif distance == "l2":
        return l2_distance
    elif distance == "l1":
        return l1_distance
    else:
        raise ValueError(f"Unknown distance function: {distance}")


def get_distance_matrix(embs, distance="cosine"):
    """Calculate distance matrix
    :param embs: embeddings
    :param distance: distance function name
    :returns: distance matrix
    """
    distance_function = get_distance_function(distance)
    n = len(embs)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = distance_function(embs[i], embs[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


def get_sorted_distance_matrix(embs, distance="cosine"):
    """Calculate sorted distance matrix
    :param embs: embeddings
    :param distance: distance function name
    :returns: sorted distance matrix
    """
    distance_matrix = get_distance_matrix(embs, distance)
    sorted_distance_matrix = {}
    for i in range(len(embs)):
        sorted_distance_matrix[i] = sorted(enumerate(distance_matrix[i]), key=lambda x: x[1])
    return sorted_distance_matrix
