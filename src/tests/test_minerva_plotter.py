import numpy as np
from sklearn.manifold import TSNE
from minerva_plotter import MinervaPlotter

def test_add_data_manual():
    mp = MinervaPlotter()
    x = np.random.randn(10, 5)
    y = np.zeros(10, dtype=int)

    mp.add_data(x, y, tag="A", marker="o")

    assert len(mp.data) == 1
    assert mp.data[0]["tag"] == "A"
    assert mp.data[0]["marker"] == "o"
    assert mp.data[0]["data_x"].shape == (10, 5)
    assert mp.data[0]["data_y"].shape == (10,)


def test_fft_application():
    mp = MinervaPlotter()
    x = np.random.randn(10, 4, 8)
    y = np.zeros(10, dtype=int)

    mp.add_data(x, y)
    mp.apply_fft(fft_axis=2)

    # FFT magnitude → same input shape
    assert mp.data[0]["data_x"].shape == (10, 4, 8)
    assert np.all(mp.data[0]["data_x"] >= 0)  # magnitude must be >= 0


def test_dimensionality_reduction():
    mp = MinervaPlotter()

    x1 = np.random.randn(20, 10)
    y1 = np.zeros(20, dtype=int)

    x2 = np.random.randn(30, 10)
    y2 = np.ones(30, dtype=int)

    mp.add_data(x1, y1)
    mp.add_data(x2, y2)

    reducer = TSNE(n_components=2, random_state=0)
    mp.apply_reducer(reducer)

    # After reduction: each sample → 2D
    assert mp.data[0]["data_x"].shape == (20, 2)
    assert mp.data[1]["data_x"].shape == (30, 2)


def test_visibility_filter():
    mp = MinervaPlotter()

    x1 = np.random.randn(10, 5)
    y1 = np.zeros(10, dtype=int)
    x2 = np.random.randn(10, 5)
    y2 = np.ones(10, dtype=int)

    mp.add_data(x1, y1, tag="A")
    mp.add_data(x2, y2, tag="B")

    mp.apply_reducer()

    # Only ensure this runs without error
    mp.scatter_plot(title="Test", visibility=[True, False])