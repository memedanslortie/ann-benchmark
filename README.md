# Approximate Nearest Neighbor Search: A Benchmark

> Master's project, Big Data course, Université Paris Cité (2025). Team of 2.

A configurable benchmarking framework comparing five **Approximate Nearest Neighbor (ANN)**
libraries on standard [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) datasets.
It measures the **recall vs. throughput** trade-off over large hyper-parameter grids and
extracts the Pareto frontier of each method.

![QPS vs Recall on SIFT-128](benchmark/visualizations/sift-128-euclidean_pareto_comparison.png)

## What it compares

| Family | Library / index | Swept hyper-parameters |
|---|---|---|
| Inverted file + product quantization | Faiss `IVFPQ` | `nlist`, `m`, `nprobe` |
| Graph (HNSW) | Faiss `HNSW` | `M`, `efConstruction`, `efSearch` |
| Graph (HNSW) | hnswlib | `M`, `ef_construction`, `ef` |
| Graph (HNSW) | Spotify Voyager | `M`, `ef_construction`, `query_ef` |
| Random-projection trees | Spotify Annoy | `n_trees`, `search_k` |

**Datasets** (HDF5, ann-benchmarks format): SIFT-128 (Euclidean), Fashion-MNIST-784
(Euclidean), NYTimes-256 (angular), Last.fm-64 (inner product).

**Metrics**: Recall@10 against exact ground truth, queries per second (QPS) and index
build time.

## Key findings

- **Graph-based indexes (HNSW variants, Voyager) dominate the high-recall regime.** On
  SIFT-128 they reach recall@10 > 0.95 while serving 10⁴ to 10⁵ queries per second.
- **IVF-PQ is the fastest at low recall** thanks to compressed codes, but it plateaus below
  perfect recall because of quantization error.
- **Annoy is the easiest to build but the slowest to query** at a given recall.
- The ranking changes with the metric and the dimensionality (see the per-dataset plots in
  [`benchmark/visualizations/`](benchmark/visualizations)).

<p align="center">
  <img src="benchmark/visualizations/algo_radar_comparison.png" width="480" alt="Radar comparison of ANN algorithms">
</p>

## Architecture

```
benchmark/
├── algos/                  # One adapter per library, sharing the same fit/query interface
│   ├── faiss_ivfpq/  faissHSNW/  hnsw/  voyager/  annoy/
├── ann_benchmark/
│   ├── datasets.py         # HDF5 loading + normalization
│   └── evaluation.py       # Recall@k
├── benchmark/
│   └── full_comparison.yaml   # Algorithms and parameter grids (declarative)
├── scripts/
│   ├── benchmark.py        # Grid-search runner (multi-process + multi-thread)
│   ├── compute_ground_truth.py  # Exact k-NN with Faiss (L2 / IP / cosine)
│   └── plot.py             # Pareto frontiers and radar charts (Plotly)
└── visualizations/         # Generated figures
```

Adding a new library only requires a small adapter class and an entry in the YAML file.

## Usage

```bash
pip install numpy h5py pyyaml tqdm matplotlib pandas plotly faiss-cpu hnswlib voyager annoy

# 1. Put the ann-benchmarks .hdf5 files in benchmark/data/, then compute the ground truth
python -m benchmark.scripts.compute_ground_truth --dataset sift-128-euclidean --metric l2

# 2. Run the grid search (all algorithms, parallel)
python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml \
       --dataset sift-128-euclidean --parallel both

# 3. Plot the Pareto frontiers
python -m benchmark.scripts.plot
```

## My contribution (Auguste Calmanovic-Plescoff)

- **Started the project**: initial Faiss indexing pipeline and repository structure.
- **Integrated three libraries**: Spotify **Voyager**, Spotify **Annoy**, and the
  graph-based **QSG-NGT**, which was evaluated and later dropped from the final comparison.
- **Data normalization** in the dataset loader, plus work on the **exact ground-truth
  computation** used to measure recall.
- **Pareto-frontier analysis**: extraction and plotting of the optimal recall/QPS
  configurations of each method.

Co-author: [Yassine Fekih](https://github.com/yassinefkh). He handled the Faiss IVF-PQ,
Faiss HNSW and hnswlib adapters, runner parallelization and the final visualizations.

## Tech stack

Python · Faiss · hnswlib · Voyager · Annoy · NumPy · HDF5 · Plotly · YAML
