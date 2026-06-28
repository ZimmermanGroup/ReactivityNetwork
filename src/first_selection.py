import networkx as nx
from dataset import *
from rmap import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import entropy, spearmanr
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class FirstBatchSelector:
    def __init__(self, target_core_ind):
        self.target_core_ind = target_core_ind
        self.dataset = SuzukiDataset()
        np.random.seed(42)
        self.test_bb_inds = np.sort(
            np.random.choice(
                np.arange(self.dataset.yield_array.shape[1]),
                14,  # Leaving 14 BBs out
                False,  # Not choosing the same building block multiple times
            )
        )
        self.train_bb_inds = [
            x for x in range(len(self.dataset.bb_smiles)) if x not in self.test_bb_inds
        ]
        self.rmap = ReactivityMap(
            source_core_inds=None,
            target_core_ind=self.target_core_ind,
            test_bb_inds=self.test_bb_inds,
            dataset=self.dataset,
        )
        (
            self.rfc_selected_dict,
            self.rfc_inspection_dict,
        ) = self.rfc_uncertainty_selection()  # Avoiding running this twice

        if not os.path.exists("figures"):
            os.mkdir("figures")
        if not os.path.exists("figures/eval_first"):
            os.mkdir("figures/eval_first")

    def _get_centroid(self, communities, adjacency_matrix):
        """Tries to get a node within the community with the highest degree as the centroid.
        Helper function to extract centroids from modularity clusters determined with networkx.

        Parameters
        ----------
        communities: list of frozenset of nodes.
            Outputs given by a nx.community function.
        adjacency_matrix: np.ndarray of shape (n_training_substrates, n_training_substrates)
            Adjacency matrix of the training substrates that was used to make the graph.

        Returns
        -------
        centroid_inds : list of inds
            Indices of the nodes in the community that are considered as centroids.
        """
        centroid_inds = []
        for community in communities:
            sub_adj_mat = adjacency_matrix[list(community), :][:, list(community)]
            degrees = np.sum(sub_adj_mat, axis=0)
            highest_degree_node = np.argmax(degrees)
            centroid_inds.append(list(community)[highest_degree_node])
        return centroid_inds

    def _unroll_arrays_for_rf(self, array):
        """Transforms the output array such that it is compatible with random forest classifiers.

        Parameters
        ----------
        array : np.ndarray
            Input or output values to rearrange for use with conventional models like RF.

        Returns
        -------
        unrolled_array : np.ndarray
            Transformed array.
        """
        if array.ndim == 2:
            return array.flatten()
        elif array.ndim == 3:
            indiv_arrays = []
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    indiv_arrays.append(array[i, j, :])
            return np.vstack(tuple(indiv_arrays))

    def _get_average_reactivity(self, centroid_inds, reactivity_array):
        """Gets the average reactivity of the centroids.

        Parameters
        ----------
        centroid_inds: list of ints or None
            Indices of the nodes in the community that are considered as centroids.
            If None, the average reactivity for the whole reacitvity array is returned.
        reactivity_array: np.ndarray of shape (n_training_substrates, )
            Reactivity array of the training substrates that was used to make the graph.

        Returns
        -------
        average_reactivities : list of floats
            Average reactivities for each centroid.
        """
        if centroid_inds is None:
            return np.mean(reactivity_array)
        else:
            return np.mean(reactivity_array[centroid_inds])

    def rmap_selection(self):
        """Returns indices of SIX BBs selected using the Reactivity Map clustering.

        Parameters
        ----------
        None

        Returns
        -------
        selected_inds : list of ints
            Indices of BBs selected to be tested in the first round.
        """
        rmap_selected_inds = self.rmap.select_first_batch(
            6, similarity_threshold=0.82  # Selecting 6 building blocks
        )
        return rmap_selected_inds

    def modularity_selection(self):
        """Returns indices of SIX BBs selected using network clustering by modularity through networkx."""
        bootstrap_similarity_matrix = self.rmap.get_similarity_matrix()
        bootstrap_adjacency_matrix = self.rmap._similarity_matrix_to_adjacency_matrix(
            bootstrap_similarity_matrix, 0.82
        )
        graph = nx.from_numpy_array(bootstrap_adjacency_matrix)
        modularity_sets = nx.community.greedy_modularity_communities(
            graph, weight="weight", best_n=6, cutoff=6
        )
        mod_selected_inds = self._get_centroid(
            modularity_sets, bootstrap_adjacency_matrix
        )
        return mod_selected_inds

    def kmeans_selection(self):
        """Conducts kmeans clustering with the descriptor array 20 times."""
        BB_desc = self.rmap.dataset.X_desc[self.target_core_ind, :, :][
            self.train_bb_inds, :
        ]
        kmeans_selected_avg_reactivities = {
            "Average reactivity class of selected BBs": []
        }
        for x in range(20):
            kmeans_dist = KMeans(n_clusters=6, random_state=42 + x).fit_transform(
                BB_desc
            )
            kmeans_selected_inds = [np.argmin(dist_row) for dist_row in kmeans_dist.T]
            kmeans_selected_avg_reactivities[
                "Average reactivity class of selected BBs"
            ].append(
                self._get_average_reactivity(
                    kmeans_selected_inds,
                    self.rmap.dataset.reactivity_array[self.target_core_ind][
                        self.train_bb_inds
                    ],
                )
            )
        return kmeans_selected_avg_reactivities

    def rfc_uncertainty_selection(self):
        """Trains a RFC with the training dataset, makes prediction on the already-seen BBs with the new core,
        selects those with the highest average entropy across the yield classes."""
        rfc_selected_avg_reactivities = {"Average reactivity class of selected BBs": []}
        all_avg_entropy = np.zeros((len(self.train_bb_inds), 20))
        n_selected = np.zeros(len(self.train_bb_inds))
        for x in range(20):
            gcv = GridSearchCV(
                RandomForestClassifier(random_state=42 + x),
                param_grid={
                    "n_estimators": [10, 20, 50, 100],
                    "max_depth": [2, 3, 5, None],
                },
                scoring="accuracy",
                n_jobs=-1,
                cv=5,
            )
            X_train = self._unroll_arrays_for_rf(
                self.rmap.dataset.X_desc[self.rmap.source_core_inds, :, :][
                    :, self.train_bb_inds, :
                ]
            )
            X_test = self.rmap.dataset.X_desc[self.target_core_ind, :, :][
                self.train_bb_inds, :
            ]
            y_train = self._unroll_arrays_for_rf(
                self.rmap.dataset.reactivity_array[self.rmap.source_core_inds, :][
                    :, self.train_bb_inds
                ]
            )
            gcv.fit(X_train, y_train)
            entropy_vals = entropy(gcv.predict_proba(X_test), axis=1)
            rfc_uncertainty_selected_inds = np.argsort(entropy_vals)[
                -1 * 6 :
            ]  # Getting highest entropy BBs
            # rfc_uncertainty_selected_inds = np.argsort(np.std(gcv.predict_proba(X_test)[:, 2]))[-1 * 6:]
            rfc_selected_avg_reactivities[
                "Average reactivity class of selected BBs"
            ].append(
                self._get_average_reactivity(
                    rfc_uncertainty_selected_inds,
                    self.rmap.dataset.reactivity_array[self.target_core_ind][
                        self.train_bb_inds
                    ],
                )
            )
            n_selected[rfc_uncertainty_selected_inds] += 1
            all_avg_entropy[:, x] = entropy_vals
        inspection_dict = {
            "Overall reactivity": np.sum(self.rmap.source_reactivity_array, axis=0),
            "Average entropy": np.mean(all_avg_entropy, axis=1),
            "Selection frequency": n_selected,
        }
        return rfc_selected_avg_reactivities, inspection_dict

    def plot_avg_reactivity(self, strategies, filename=None):
        """Plots the average reactivity of BBs selected through different strategies.

        Parameters
        ----------
        strategies : list of str
            Methods to plot.
        filename : None or str
            Filename to save the resulting plot.

        Returns
        -------
        None
        """
        colors = sns.color_palette("colorblind", 8)
        color_dict = {
            "Actual": colors[4],
            "RMap": colors[1],
            "Descriptor KMeans": colors[2],
            "RFC Uncertainty": colors[0],
            "Network Modularity": colors[-1],
        }
        inds_to_plot = {}
        dicts_to_plot = {}
        for strategy in strategies:
            if strategy == "rmap":
                inds_to_plot.update({"RMap": self.rmap_selection()})
            elif strategy == "modularity":
                inds_to_plot.update({"Network Modularity": self.modularity_selection()})
            elif strategy == "kmeans":
                dicts_to_plot.update({"Descriptor KMeans": self.kmeans_selection()})
            elif strategy == "rfc":
                dicts_to_plot.update({"RFC Uncertainty": self.rfc_selected_dict})
            else:
                raise ValueError(
                    "Strategy to plot must be one of the following: 'rmap', 'modularity', 'kmeans', 'rfc'."
                )

        fig, ax = plt.subplots(figsize=(2.45, 1.63))
        colors = sns.color_palette("colorblind", 5)
        if len(dicts_to_plot.keys()) > 0:
            for k, v in dicts_to_plot.items():
                sns.kdeplot(
                    data=v,
                    x="Average reactivity class of selected BBs",
                    ax=ax,
                    color=color_dict[k],
                    ls="--",
                )
                ax.axvline(
                    x=np.mean(v["Average reactivity class of selected BBs"]),
                    ymin=0,
                    ymax=1,
                    color=color_dict[k],
                    label=k,
                )  # linestyle='--',
        previous_avg_vals = []  # To prevent overlap between vertical lines
        for k, v in inds_to_plot.items():
            avg_reactivity = self._get_average_reactivity(
                v,
                self.rmap.dataset.reactivity_array[self.target_core_ind][
                    self.train_bb_inds
                ],
            )
            # print(k, avg_reactivity)
            if avg_reactivity in previous_avg_vals:
                avg_reactivity += 0.05
            if avg_reactivity == 2:
                avg_reactivity = 1.95
            previous_avg_vals.append(avg_reactivity)
            ax.axvline(
                x=avg_reactivity, ymin=0, ymax=1, color=color_dict[k], label=k
            )  # linestyle='--',
        true_avg_reactivity = self._get_average_reactivity(
            None,
            self.rmap.dataset.reactivity_array[self.target_core_ind][
                self.train_bb_inds
            ],
        )
        if true_avg_reactivity in previous_avg_vals:
            true_avg_reactivity += 0.05
        ax.axvline(
            x=true_avg_reactivity,
            ymin=0,
            ymax=1,
            color=color_dict["Actual"],
            label="Actual",
        )

        if self.target_core_ind < 9:
            loc = "upper left"
        else:
            loc = "upper right"
        ax.legend(prop=fm.FontProperties(family="Arial", size=7), loc=loc)
        ax.set_xlim(0, 2)
        ax.set_xticks(np.arange(0, 2.1, 0.5))
        ax.set_xticklabels(
            np.arange(0, 2.1, 0.5), fontdict={"fontsize": 7, "fontfamily": "Arial"}
        )
        ax.set_yticklabels([])
        ax.set_ylabel(ax.get_ylabel(), fontdict={"fontsize": 8, "fontfamily": "Arial"})
        ax.set_xlabel(ax.get_xlabel(), fontdict={"fontsize": 8, "fontfamily": "Arial"})
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.5)
        if filename is None:
            plt.show()
        else:
            plt.savefig(f"figures/eval_first/{filename}.svg", dpi=300, format="svg")

    def plot_overall_reactivity_vs_avg_entropy(self, filename=None):
        fig, ax = plt.subplots(figsize=(3.5, 2))
        sns.scatterplot(
            self.rfc_inspection_dict,
            x="Average entropy",
            y="Overall reactivity",
            hue="Selection frequency",
            palette="crest",
            legend=False,
        )
        max_freq_val = self.rfc_inspection_dict["Selection frequency"].max()
        norm = plt.Normalize(
            self.rfc_inspection_dict["Selection frequency"].min(), max_freq_val
        )
        sm = cm.ScalarMappable(cmap="crest", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        if max_freq_val > 15:
            max_cbar = 16
        else:
            max_cbar = 11
        cbar.ax.set_yticks(np.arange(0, max_cbar, 5))
        cbar.ax.set_yticklabels(
            np.arange(0, max_cbar, 5), fontdict={"fontsize": 7, "fontfamily": "Arial"}
        )
        cbar.set_label(
            "Selection frequency",
            fontdict={"fontsize": 8, "fontfamily": "Arial"},
            labelpad=5,
            y=0.5,
        )
        res = spearmanr(
            self.rfc_inspection_dict["Average entropy"],
            self.rfc_inspection_dict["Overall reactivity"],
            alternative="greater",
        )
        ax.text(
            x=0.2,
            y=0.75,
            s=f"Spearman={round(res.statistic, 3):.3f}",
            transform=ax.transAxes,
            fontdict={"fontsize": 7, "fontfamily": "Arial"},
        )
        ax.set_ylim(0, 17)
        ax.set_yticklabels(
            np.arange(0, 16, 5), fontdict={"fontsize": 7, "fontfamily": "Arial"}
        )
        ax.set_ylabel(ax.get_ylabel(), fontdict={"fontsize": 8, "fontfamily": "Arial"})
        ax.set_xlabel(ax.get_xlabel(), fontdict={"fontsize": 8, "fontfamily": "Arial"})
        ax.set_xticklabels(
            ax.get_xticklabels(), fontdict={"fontsize": 7, "fontfamily": "Arial"}
        )
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.5)
        if filename is None:
            plt.show()
        else:
            plt.savefig(f"figures/eval_first/{filename}.svg", dpi=300, format="svg")


if __name__ == "__main__":
    context = input("Where in the manuscript will the figures go? ")
    if context == "main":
        strategies = ["rmap", "rfc", "kmeans"]
        cores = [3, 7, 10]
        fignum = "Figure5"
    elif context == "si":
        strategies = ["rmap", "modularity", "rfc", "kmeans"]
        cores = [0, 1, 3, 7, 8, 9, 10, 12]
        fignum = "FigureS18"
    else:
        raise ValueError("Entered value must be either main or si.")

    for core in cores:
        first_selection = FirstBatchSelector(core)
        first_selection.plot_avg_reactivity(
            strategies, f"{fignum}_first_selection_core_{core+1}"
        )
        if context == "si":
            first_selection.plot_overall_reactivity_vs_avg_entropy(
                f"FigureS19_uncertainty_vs_reactivity_{core+1}"
            )
