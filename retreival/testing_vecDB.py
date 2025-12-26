"""Realistic FAISS HNSW Benchmark - FIXED clustering that survives normalization"""

import faiss
import numpy as np
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Store benchmark metrics"""

    index_type: str
    build_time: float
    search_time: float
    recall_at_k: float
    mean_similarity: float
    memory_mb: float
    queries_per_second: float


class RealisticVectorBenchmark:
    """Benchmark with PROPERLY clustered embeddings"""

    def __init__(self, dimension: int = 384, num_vectors: int = 10000):
        self.dimension = dimension
        self.num_vectors = num_vectors
        self.num_queries = 100

        np.random.seed(42)
        self.vectors, self.query_vectors = self._generate_realistic_embeddings()

    def _generate_realistic_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate vectors with STRONG cluster structure that survives normalization.

        Key insight: In high dimensions, adding small noise then normalizing
        destroys clustering. Instead, we use SLERP (spherical interpolation)
        to stay close to the cluster center on the unit sphere.
        """
        # Create strong clusters
        num_clusters = max(100, self.num_vectors // 100)
        vectors_per_cluster = self.num_vectors // num_clusters

        all_vectors = []
        cluster_centers = []

        print(
            f"Generating {num_clusters} clusters with ~{vectors_per_cluster} vectors each..."
        )

        for i in range(num_clusters):
            # Create cluster center (unit vector)
            center = np.random.randn(self.dimension).astype("float32")
            center = center / np.linalg.norm(center)
            cluster_centers.append(center)

            # Generate vectors CLOSE to center using spherical interpolation
            # This maintains high cosine similarity even after normalization
            cluster_vectors = []
            for _ in range(vectors_per_cluster):
                # Create a random direction
                random_vec = np.random.randn(self.dimension).astype("float32")
                random_vec = random_vec / np.linalg.norm(random_vec)

                # SLERP: interpolate between center and random direction
                # alpha close to 1.0 = stay very close to center
                # This creates cosine similarities of 0.8-0.95 within cluster
                alpha = np.random.uniform(0.85, 0.95)

                # Spherical interpolation
                dot = np.dot(center, random_vec)
                dot = np.clip(dot, -1.0, 1.0)
                theta = np.arccos(dot)

                if theta < 0.001:  # Vectors are already very close
                    vec = center
                else:
                    vec = (np.sin((1 - alpha) * theta) / np.sin(theta)) * center + (
                        np.sin(alpha * theta) / np.sin(theta)
                    ) * random_vec
                    vec = vec / np.linalg.norm(vec)

                cluster_vectors.append(vec)

            all_vectors.extend(cluster_vectors)

        # Handle remainder
        remainder = self.num_vectors - len(all_vectors)
        if remainder > 0:
            center = np.random.randn(self.dimension).astype("float32")
            center = center / np.linalg.norm(center)
            for _ in range(remainder):
                random_vec = np.random.randn(self.dimension).astype("float32")
                random_vec = random_vec / np.linalg.norm(random_vec)
                alpha = np.random.uniform(0.85, 0.95)
                dot = np.dot(center, random_vec)
                dot = np.clip(dot, -1.0, 1.0)
                theta = np.arccos(dot)
                if theta < 0.001:
                    vec = center
                else:
                    vec = (np.sin((1 - alpha) * theta) / np.sin(theta)) * center + (
                        np.sin(alpha * theta) / np.sin(theta)
                    ) * random_vec
                    vec = vec / np.linalg.norm(vec)
                all_vectors.append(vec)

        vectors = np.array(all_vectors, dtype="float32")

        # Generate queries close to random cluster centers
        query_vectors = []
        for _ in range(self.num_queries):
            # Pick a random cluster center
            center = cluster_centers[np.random.randint(len(cluster_centers))]

            # Create query close to this center (alpha = 0.75-0.90)
            # Queries are less similar than cluster members (realistic!)
            random_vec = np.random.randn(self.dimension).astype("float32")
            random_vec = random_vec / np.linalg.norm(random_vec)
            alpha = np.random.uniform(0.75, 0.90)

            dot = np.dot(center, random_vec)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            if theta < 0.001:
                query = center
            else:
                query = (np.sin((1 - alpha) * theta) / np.sin(theta)) * center + (
                    np.sin(alpha * theta) / np.sin(theta)
                ) * random_vec
                query = query / np.linalg.norm(query)

            query_vectors.append(query)

        query_vectors = np.array(query_vectors, dtype="float32")

        # Diagnostic: Verify clustering worked
        sample_sims = np.dot(vectors[:1000], query_vectors[:10].T)
        print(
            f"✓ Similarity range: min={sample_sims.min():.3f}, "
            f"mean={sample_sims.mean():.3f}, median={np.median(sample_sims):.3f}, max={sample_sims.max():.3f}"
        )

        # Check that we have actual clusters
        first_query_sims = np.dot(vectors, query_vectors[0])
        top10_sims = np.sort(first_query_sims)[-10:]
        print(
            f"✓ First query top-10 similarities: {top10_sims.min():.3f} to {top10_sims.max():.3f}"
        )

        if sample_sims.max() < 0.5:
            print(
                "⚠ WARNING: Similarities too low! Clusters may not have formed properly."
            )
        else:
            print(f"✓ Strong clustering detected (max sim = {sample_sims.max():.3f})")

        return vectors, query_vectors

    def build_flat_index(self) -> Tuple[faiss.Index, float]:
        """Build exact search index"""
        start = time.time()
        index = faiss.IndexFlatIP(self.dimension)
        index.add(self.vectors)
        build_time = time.time() - start
        return index, build_time

    def build_hnsw_index(
        self, m: int = 32, ef_construction: int = 200
    ) -> Tuple[faiss.Index, float]:
        """Build HNSW index"""
        start = time.time()
        index = faiss.IndexHNSWFlat(self.dimension, m)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = ef_construction
        index.add(self.vectors)
        build_time = time.time() - start
        return index, build_time

    def measure_search(
        self, index: faiss.Index, k: int = 10, ef_search: int = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Measure search performance"""
        if ef_search and hasattr(index, "hnsw"):
            index.hnsw.efSearch = ef_search

        start = time.time()
        distances, indices = index.search(self.query_vectors, k)
        search_time = time.time() - start

        return distances, indices, search_time

    def calculate_recall(
        self, ground_truth: np.ndarray, test_results: np.ndarray, k: int = 10
    ) -> float:
        """Calculate recall@k"""
        recalls = []
        for gt, test in zip(ground_truth, test_results):
            intersection = len(set(gt[:k]) & set(test[:k]))
            recalls.append(intersection / k)
        return np.mean(recalls)

    def run_benchmark(self, k: int = 10) -> List[BenchmarkResult]:
        """Run complete benchmark"""
        results = []

        # Exact search baseline
        print("\nBuilding Flat index...")
        flat_index, flat_build = self.build_flat_index()
        flat_dists, flat_indices, flat_search = self.measure_search(flat_index, k)

        results.append(
            BenchmarkResult(
                index_type="IndexFlatIP (Exact)",
                build_time=flat_build,
                search_time=flat_search,
                recall_at_k=1.0,
                mean_similarity=flat_dists.mean(),
                memory_mb=self.num_vectors * self.dimension * 4 / (1024**2),
                queries_per_second=self.num_queries / flat_search,
            )
        )

        # HNSW with realistic production configs
        hnsw_configs = [
            {"m": 16, "ef_construction": 100, "ef_search": 16},
            {"m": 16, "ef_construction": 100, "ef_search": 48},
            {"m": 32, "ef_construction": 200, "ef_search": 48},
            {"m": 32, "ef_construction": 200, "ef_search": 128},
            {"m": 32, "ef_construction": 200, "ef_search": 256},
            {"m": 32, "ef_construction": 400, "ef_search": 256},
            {"m": 64, "ef_construction": 400, "ef_search": 256},
            {"m": 64, "ef_construction": 400, "ef_search": 512},
        ]

        for config in hnsw_configs:
            print(
                f"Testing HNSW M={config['m']}, efC={config['ef_construction']}, efS={config['ef_search']}..."
            )
            hnsw_index, hnsw_build = self.build_hnsw_index(
                config["m"], config["ef_construction"]
            )
            hnsw_dists, hnsw_indices, hnsw_search = self.measure_search(
                hnsw_index, k, config["ef_search"]
            )

            recall = self.calculate_recall(flat_indices, hnsw_indices, k)

            results.append(
                BenchmarkResult(
                    index_type=f"HNSW (M={config['m']}, efC={config['ef_construction']}, efS={config['ef_search']})",
                    build_time=hnsw_build,
                    search_time=hnsw_search,
                    recall_at_k=recall,
                    mean_similarity=hnsw_dists.mean(),
                    memory_mb=(
                        self.num_vectors * self.dimension * 4
                        + self.num_vectors * config["m"] * 8
                    )
                    / (1024**2),
                    queries_per_second=self.num_queries / hnsw_search,
                )
            )

        return results

    def print_results(self, results: List[BenchmarkResult]):
        """Print formatted results"""
        print("\n" + "=" * 120)
        print(
            f"BENCHMARK: {self.num_vectors:,} vectors, {self.dimension}D, {self.num_queries} queries"
        )
        print("=" * 120)
        print(
            f"{'Index':<45} {'Build(s)':<10} {'Search(s)':<10} {'QPS':<10} {'Recall@10':<12} {'Speedup':<10} {'Mem(MB)':<10}"
        )
        print("-" * 120)

        baseline_time = results[0].search_time

        for r in results:
            speedup = baseline_time / r.search_time if r.search_time > 0 else 0
            print(
                f"{r.index_type:<45} {r.build_time:<10.4f} {r.search_time:<10.6f} "
                f"{r.queries_per_second:<10.0f} {r.recall_at_k:<12.3f} "
                f"{speedup:<10.2f}x {r.memory_mb:<10.1f}"
            )

        print("=" * 120)

        # Analysis
        print("\n📊 PERFORMANCE RECOMMENDATIONS:")
        best_configs = []
        for r in results[1:]:
            if r.recall_at_k >= 0.95:
                speedup = baseline_time / r.search_time
                print(
                    f"✓ {r.index_type}: {speedup:.1f}x speedup, {r.recall_at_k:.1%} recall - EXCELLENT"
                )
                best_configs.append(r)
            elif r.recall_at_k >= 0.90:
                speedup = baseline_time / r.search_time
                print(
                    f"○ {r.index_type}: {speedup:.1f}x speedup, {r.recall_at_k:.1%} recall - GOOD"
                )
            elif r.recall_at_k >= 0.80:
                print(
                    f"△ {r.index_type}: {r.recall_at_k:.1%} recall - OK for speed-critical apps"
                )
            else:
                print(f"✗ {r.index_type}: {r.recall_at_k:.1%} recall - Too low")

        if best_configs:
            print(f"\n🎯 RECOMMENDED: {best_configs[0].index_type}")


if __name__ == "__main__":
    print("\n" + "=" * 120)
    print("REALISTIC HNSW BENCHMARK - Proper Spherical Clustering")
    print("=" * 120)

    # Small scale
    print("\n[TEST 1: 10K vectors - Development scale]")
    bench_small = RealisticVectorBenchmark(dimension=384, num_vectors=10000)
    results_small = bench_small.run_benchmark(k=10)
    bench_small.print_results(results_small)

    # Medium scale
    print("\n\n[TEST 2: 100K vectors - Production scale]")
    bench_medium = RealisticVectorBenchmark(dimension=384, num_vectors=100000)
    results_medium = bench_medium.run_benchmark(k=10)
    bench_medium.print_results(results_medium)

    print("\n" + "=" * 120)
    print("💡 KEY INSIGHTS FOR PRODUCTION:")
    print("1. Real embeddings form clusters on the unit hypersphere")
    print("2. Within-cluster similarities: 0.8-0.95 (documents about same topic)")
    print("3. Query-to-document similarities: 0.6-0.9 (semantic relevance)")
    print("4. HNSW navigates these clusters efficiently")
    print("5. Typical production config: M=32, efConstruction=200, efSearch=100-200")
    print("\n⚠️  ALWAYS validate with YOUR embedding model - this is a simulation!")
    print("=" * 120)
