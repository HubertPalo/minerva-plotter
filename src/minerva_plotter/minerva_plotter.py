from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


class MinervaPlotter:
    """
    A utility class for collecting, transforming, reducing, and visualizing
    datasets using scatter plots.

    This class is designed to simplify the workflow for comparing multiple datasets
    (e.g., real vs. synthetic, multiple encoders, different SSL models).
    It supports:
      - different data sources (raw data or Lightning datamodules),
      - optional FFT preprocessing,
      - shared dimensionality reduction (e.g., t-SNE),
      - labeled scatter visualization.

    Attributes
    ----------
    data : list of dict
        Internal list storing each added dataset with keys:
        `'data_x'`, `'data_y'`, `'tag'`, and `'marker'`.
    """

    def __init__(self):
        self.data = []

    # -------------------------------------------------------------------------
    # DEFAULT LABELS & COLORS
    # -------------------------------------------------------------------------

    @property
    def default_labels_dict(self):
        """dict: From class IDs to human-readable activity labels."""
        return {
            0: "SIT",
            1: "STAND",
            2: "WALK",
            3: "STAIR-UP",
            4: "STAIR-DOWN",
            5: "RUN",
            -1000: "SYNTHETIC",
        }

    @property
    def default_color_dict(self):
        """dict: From class IDs to default matplotlib colors."""
        return {
            0: "blue",
            1: "orange",
            2: "green",
            3: "purple",
            4: "red",
            5: "brown",
            -1000: "lightgray",
        }
    
    # -------------------------------------------------------------------------
    # DATA INGESTION
    # -------------------------------------------------------------------------

    def add_data(
        self,
        data_x=None,
        data_y=None,
        datamodule=None,
        datamodule_partition="train",
        tag=None,
        marker=None,
    ):
        """
        Add a dataset to the plotter.

        Parameters
        ----------
        data_x : ndarray, optional
            Input data of shape (N, ...) representing samples.
        data_y : ndarray, optional
            Integer labels of shape (N,).
        datamodule : LightningDataModule, optional
            If provided, data will be loaded from its train/test dataloader.
        datamodule_partition : str, default="train"
            Either "train" or "test". Determines which dataloader to extract from.
        tag : str, optional
            Label prefix added to legend entries for this dataset.
        marker : str, optional
            Matplotlib marker style used for scatter points.

        Notes
        -----
        If `datamodule` is provided, it overrides `data_x` and `data_y`.
        """
        if datamodule is not None:
            if datamodule_partition == "train":
                datamodule.setup(stage="fit")
                data_x, data_y = datamodule.train_dataloader().dataset[:]
            elif datamodule_partition == "test":
                datamodule.setup(stage="test")
                data_x, data_y = datamodule.test_dataloader().dataset[:]
        data_unit = {"data_x": data_x, "data_y": data_y, "tag": tag, "marker": marker}
        self.data.append(data_unit)
    
    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------

    def apply_fft(self, fft_axis=2):
        """
        Apply FFT along a given axis for all datasets ingested.

        Parameters
        ----------
        fft_axis : int, default=2
            Axis along which FFT is computed. Its default value (2) is suggested for
            Minerva datasets, which deliver data in shape (N,6,60)

        Notes
        -----
        Replaces each dataset's `data_x` with the absolute value of FFT.
        """
        for data_unit in self.data:
            data_unit["data_x"] = np.fft.fft(data_unit["data_x"], axis=fft_axis)
            data_unit["data_x"] = np.abs(data_unit["data_x"])
    
    # -------------------------------------------------------------------------
    # DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------------------

    def apply_reducer(self, reducer=TSNE(n_components=2, random_state=42)):
        """
        Apply a dimensionality reduction method (by default, t-SNE) jointly to all
        datasets.

        Parameters
        ----------
        reducer : sklearn transformer, default=TSNE(...)
            Any object with a `.fit_transform()` method (PCA, UMAP, t-SNE, etc.)

        Notes
        -----
        - All datasets are concatenated before reduction.
        - Saves the 2D embeddings back into each dataset in-place.
        - Ensures the shared embedding space remains consistent.
        """
        whole_data = np.concatenate([du["data_x"] for du in self.data], axis=0)
        whole_data_2d = whole_data.reshape(whole_data.shape[0], -1)
        whole_data_2d = reducer.fit_transform(whole_data_2d)
        start_idx = 0
        for data_unit in self.data:
            end_idx = start_idx + data_unit["data_x"].shape[0]
            data_unit["data_x"] = whole_data_2d[start_idx:end_idx]
            start_idx = end_idx

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------

    def scatter_plot(
        self,
        title: str = None,
        visibility: list = None,
        kwargs={},
        filename=None,
        savefig_kwargs={},
        legend_loc="simple",
        legend_cols=1,
    ):
        """
        Generates a 2D scatter plot including all collected datasets.

        Parameters
        ----------
        title : str, optional
            Figure title.
        visibility : list of bool, optional
            If provided, only datasets with True will be plotted.
        kwargs : dict, optional
            Additional arguments forwarded to `plt.scatter()`.
        filename : str, optional
            If provided, saves the figure to this path.
        savefig_kwargs : dict, optional
            Extra parameters passed to `plt.savefig()`.
        legend_loc : {"simple", "out-lower-center", "out-right-center"}
            Determines legend placement. If "simple", the best placement will be
            determined by matplotlib. 
        legend_cols : int, default=1
            Number of legend columns.

        Notes
        -----
        - Each dataset may have a `tag` and a custom `marker`, defined when collected.
        - Colors and labels default to `default_color_dict` and `default_labels_dict`.
        """
        plt.figure(figsize=(8, 6))

        for data_idx, data_unit in enumerate(self.data):
            if visibility is not None and not visibility[data_idx]:
                continue
            data_x = data_unit["data_x"]
            data_y = data_unit["data_y"]

        for data_idx, data_unit in enumerate(self.data):
            if visibility is not None and not visibility[data_idx]:
                continue
            data_x = data_unit["data_x"]
            data_y = data_unit["data_y"]
            tag = data_unit.get("tag", None)
            marker = data_unit.get("marker", "o")
            for unique_id in np.unique(data_y):
                color = self.default_color_dict.get(unique_id, "black")
                label = self.default_labels_dict.get(unique_id, str(unique_id))
                if tag is not None:
                    label = f"{tag} - {label}"
                plt.scatter(
                    data_x[data_y == unique_id, 0],
                    data_x[data_y == unique_id, 1],
                    label=label,
                    color=color,
                    marker=marker,
                    **kwargs,
                )
        plt.title(title)
        if legend_loc == "simple":
            plt.legend(ncol=legend_cols)
        elif legend_loc == "out-lower-center":
            plt.legend(loc="upper center", bbox_to_anchor=(0.50, -0), ncol=legend_cols)
        elif legend_loc == "out-right-center":
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=legend_cols)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        if filename:
            plt.savefig(filename, **savefig_kwargs)
        plt.show()
