from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorDatabase:
    def __init__(self):
        self.embeddings = []  # List to store embeddings
        self.documents = []   # List to store corresponding documents

    def add_document(self, document: str, embedding: List[float]):
        """Add a document and its corresponding embedding to the database."""
        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Search for the top_k most similar documents to the query embedding."""
        if not self.embeddings:
            return []

        # Convert embeddings to numpy array for cosine similarity calculation
        embeddings_array = np.array(self.embeddings)
        query_array = np.array(query_embedding).reshape(1, -1)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_array, embeddings_array).flatten()

        # Get the indices of the top_k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Return the top_k most similar documents
        return [self.documents[i] for i in top_indices]