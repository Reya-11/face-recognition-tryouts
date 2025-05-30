import numpy as np

memory = {}

def add_to_memory(name, embedding):
    memory[name] = embedding

def search_memory(new_embedding, threshold=0.5):
    from numpy.linalg import norm
    for name, ref in memory.items():
        sim = np.dot(new_embedding, ref) / (norm(new_embedding) * norm(ref))
        if sim > threshold:
            return name, sim
    return None, 0
