import numpy as np
from typing import List
from sklearn.cluster import HDBSCAN, KMeans

from models.document_chunk import DocumentChunk


class SuperChunk:
    """Represents a cluster of merged chunks"""

    def __init__(self, merged_chunks: List[DocumentChunk], superchunk_id: int):
        self.superchunk_id = superchunk_id
        self.merged_chunks = sorted(merged_chunks, key=lambda x: x.chunk_id)
        self.text = "\n\n".join([chunk.text for chunk in self.merged_chunks])

        # Preserve metadata from constituent chunks
        self.doc_name = self.merged_chunks[0].doc_name

        # Get all unique page numbers as a list
        self.page_numbers = sorted(
            list(set(chunk.page_num for chunk in self.merged_chunks))
        )

        self.chunk_ids = [chunk.chunk_id for chunk in self.merged_chunks]

    def __repr__(self):
        return f"SuperChunk(id={self.superchunk_id}, chunks={len(self.merged_chunks)}, pages={self.page_numbers})"


class ChunkClusterer:
    """Handles semantic clustering of chunks into super-chunks using HDBSCAN"""

    def __init__(self, max_cluster_size: int = 5, min_cluster_size: int = 2):
        """
        Initialize HDBSCAN clusterer

        Args:
            max_cluster_size: Maximum chunks per super-chunk (hard limit)
            min_cluster_size: Minimum chunks to form a cluster
        """
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min(min_cluster_size, max_cluster_size)

    def create_superchunks(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> List[SuperChunk]:
        """
        Create super-chunks using HDBSCAN clustering

        Args:
            chunks: List of base chunks
            embeddings: Embeddings for each chunk (normalized)

        Returns:
            List of SuperChunk objects
        """
        if len(chunks) < self.min_cluster_size:
            return [SuperChunk(chunks, 0)]

        # HDBSCAN clustering with euclidean metric on normalized embeddings
        # This is mathematically equivalent to cosine distance
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            metric="euclidean",  # Use euclidean on normalized vectors
            cluster_selection_method="eom",
            allow_single_cluster=True,
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        # Enforce max cluster size constraint
        cluster_labels = self._enforce_max_size(cluster_labels, embeddings)

        # Create super-chunks from clusters
        return self._create_superchunks_from_labels(chunks, cluster_labels)

    def _enforce_max_size(
        self, labels: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Split clusters that exceed max_cluster_size using KMeans sub-clustering

        Args:
            labels: Cluster assignments
            embeddings: Normalized embeddings

        Returns:
            Updated labels with size constraints enforced
        """

        unique_labels = np.unique(labels)
        new_labels = labels.copy()
        next_label = labels.max() + 1

        for label in unique_labels:
            mask = labels == label
            cluster_size = mask.sum()

            if cluster_size > self.max_cluster_size:
                cluster_indices = np.where(mask)[0]
                cluster_embeddings = embeddings[mask]

                # Calculate number of sub-clusters needed
                n_subclusters = int(np.ceil(cluster_size / self.max_cluster_size))

                # Use KMeans for sub-clustering
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
                sub_labels = kmeans.fit_predict(cluster_embeddings)

                # Assign new labels
                for i, sub_label in enumerate(sub_labels):
                    if sub_label == 0:
                        continue
                    else:
                        new_labels[cluster_indices[i]] = next_label + sub_label - 1

                next_label += n_subclusters - 1

        return new_labels

    def _create_superchunks_from_labels(
        self, chunks: List[DocumentChunk], labels: np.ndarray
    ) -> List[SuperChunk]:
        """
        Create SuperChunk objects from cluster labels

        Args:
            chunks: List of chunks
            labels: Cluster assignment for each chunk

        Returns:
            List of SuperChunk objects
        """
        # Group chunks by cluster label
        clusters = {}
        for chunk, label in zip(chunks, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunk)

        # Create super-chunks
        superchunks = []
        for superchunk_id, (label, cluster_chunks) in enumerate(
            sorted(clusters.items())
        ):
            superchunk = SuperChunk(cluster_chunks, superchunk_id)
            superchunks.append(superchunk)

        return superchunks
