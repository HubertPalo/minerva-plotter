# MinervaPlotter

MinervaPlotter is a small utility class for visualizing datasets in a shared
low-dimensional space. It simplifies a common workflow used in representation
learning and SSL research:

1. Load one or more datasets  
2. (Optional) Apply transformations such as FFT  
3. Reduce dimensionality jointly (e.g., t-SNE, PCA, UMAP)  
4. Produce high-quality 2D scatter plots with labels and colors  

MinervaPlotter is particularly useful when comparing:
- Real vs. synthetic samples  
- Different encoders or SSL methods  
- Feature distributions across datasets  
- Data before/after preprocessing  

---

## Features

- ✅ Add data manually or load from PyTorch Lightning `DataModule`s  
- ✅ Optional FFT preprocessing  
- ✅ Dimensionality reduction using any sklearn transformer  
- ✅ Consistent joint embedding across all datasets  
- ✅ Labeled scatter plots with customizable markers and legend placement  

---

## Basic Usage

``` python
from minerva_plotter import MinervaPlotter
from sklearn.manifold import TSNE
import numpy as np

# Create some toy data
x1 = np.random.randn(100, 10)
y1 = np.zeros(100)

x2 = np.random.randn(100, 10)
y2 = np.ones(100)

# Create plotter
mp = MinervaPlotter()

# Add data
mp.add_data(x1, y1, tag="Dataset A", marker="o")
mp.add_data(x2, y2, tag="Dataset B", marker="^")

# Dimensionality reduction (t-SNE)
mp.apply_reducer(TSNE(n_components=2, random_state=42))

# Visualize
mp.scatter_plot(title="MinervaPlotter Example")
```

---

## Saving Figures

``` python
mp.scatter_plot(
    title="Embedding",
    filename="embedding.png",
    savefig_kwargs={"dpi": 300}
)
```