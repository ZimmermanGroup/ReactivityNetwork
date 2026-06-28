import argparse
from dataset import *
from rmap import *
from tqdm import tqdm
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import joblib
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from gcn import *
import networkx as nx
from scipy.stats import entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Specify the evaluation to run.")
    parser.add_argument(
        "--feature",
        default="desc",
        choices=["desc", "fp", "ohe"],
        type=str,
        help="What type of feature to use for modeling.",
    )
    parser.add_argument(
        "--target_core",
        type=int,
        help="Which target core to evaluate.",
    )
    parser.add_argument(
        "--n1",
        default=6,
        type=int,
        help="Number of boronics to sample for the first batch with the target halide.",
    )
    parser.add_argument(
        "--n1_strategy",
        default="rmap",
        choices=[
            "rmap",
            "modularity",
            "uncertainty",
            "desc_cluster",
            "random",
            "greedy",
        ],
        help="How to select the first batch of BBs to try with the new core.",
    )
    parser.add_argument(
        "--n1_random",
        default=100,
        type=int,
        help="How many random selections to try for the first batch.",
    )
    parser.add_argument(
        "--n_test",
        default=14,
        type=int,
        help="Number of boronics to leave out for each bootstrap sample.",
    )
    parser.add_argument(
        "--n_bootstrap",
        default=20,
        type=int,
        help="Number of evaluations to make by leaving out different BBs.",
    )
    parser.add_argument(
        "--model",
        choices=["rmap", "rfc_combined", "rfc_target", "gcn", "baseline", "knn", "svm"],
        help="Which algorithm to use.",
    )
    parser.add_argument(
        "--subgraph",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether to use highly reactive cores in the training dataset to build Reactivity Map for LP.",
    )
    parser.add_argument(
        "--tanimoto",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether to use Tanimoto similarity, instead of reactivity similarity, to build the Reactivity Map for LP.",
    )
    args = parser.parse_args()
    return args


class Evaluator:
    """Evaluates the quality of the prediction on remaining BBs after the obtaining outcomes with the first batch."""

    def __init__(
        self,
        parser,
    ):
        self.parser = parser
        self.dataset = SuzukiDataset()
        if self.parser.feature == "desc":
            self.result_filename = f"saved_results/{self.dataset}/{self.parser.target_core}/{self.parser.n1_strategy}_{self.parser.n1}_{self.parser.model}"
        else:
            self.result_filename = f"saved_results/{self.dataset}/{self.parser.target_core}/{self.parser.n1_strategy}_{self.parser.n1}_{self.parser.model}_{self.parser.feature}"
        if self.parser.subgraph:
            self.result_filename += "_subgraph"
        elif self.parser.tanimoto :
            self.result_filename += "_tanimoto"
        if self.parser.n_test != 14 :
            self.result_filename += f"_ntest{self.parser.n_test}"
        if self.parser.n_bootstrap != 20 :
            self.result_filename += f"_nbootstrap{self.parser.n_bootstrap}"
        self.result_filename += ".joblib"
        self._mkdir("saved_results")
        self._mkdir(f"saved_results/{self.dataset}")
        self._mkdir(f"saved_results/{self.dataset}/{self.parser.target_core}")

        self.target_core_ind = self.parser.target_core
        self.source_core_inds = [
            i for i in range(len(self.dataset.core_smiles)) if i != self.target_core_ind
        ]
        # Defining the positive class that we should find for different target cores.
        if self.target_core_ind in [0, 1, 3, 7, 8]:
            self.desired_class = 2
        else:
            self.desired_class = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def _split_dataset(self):
        """Splits datasets into source and target, to use in downstream evaluations.
        NOTE THAT ARRAYS THAT ARE RESULTS OF THIS SPLITTING HAS BOTH TRAINING AND TEST BBs

        Parameters
        ----------
        None

        Returns
        -------
        (source_X, source_reactivity_array, source_yield_array), (target_X, target_reactivity_array, target_yield_array)
            Array to use as input, reactivity array (targets for multi-class classification) and yield array (targets for regression)
        """
        source_reactivity_array = self.dataset.reactivity_array[self.source_core_inds]
        source_yield_array = self.dataset.yield_array[self.source_core_inds]
        target_reactivity_array = self.dataset.reactivity_array[self.target_core_ind]
        target_yield_array = self.dataset.yield_array[self.target_core_ind]
        if self.parser.feature == "fp":
            source_X = self.dataset.X_fp[self.source_core_inds, :, :]
            target_X = self.dataset.X_fp[self.target_core_ind, :, :]
        elif self.parser.feature == "desc":
            source_X = self.dataset.X_desc[self.source_core_inds, :, :]
            target_X = self.dataset.X_desc[self.target_core_ind, :, :]
        elif self.parser.feature == "ohe":
            source_X = self.dataset.X_ohe[self.source_core_inds, :, :]
            target_X = self.dataset.X_ohe[self.target_core_ind, :, :]
        return (
            (source_X, source_reactivity_array, source_yield_array),
            (target_X, target_reactivity_array, target_yield_array),
        )

    def evaluate(self):
        """Conducts one evaluation for a specific target core."""
        source_array_tuple, target_array_tuple = self._split_dataset()
        results = {
            "bootstrap_id": [],  # Bootstrap iteration
            "selection_id": [],  # Selection iteration within a bootstrap (for different first batch selections; used for random selections)
            "train_bb_inds": [],  # 55 BB indices used for training in each bootstrap
            "test_bb_inds": [],  # 14 BB indices used for testing in each bootstrap
            "proba": [],  # For precision-recall curve visualization purposes; predicted probability of desired reactivity class for remaining BBs
            "target": [],  # Binary reactivity classes for the remaining BBs
            "Number of desired reactivity class in first batch": [],
            "remaining_inds": [],  # Indices within the train_bb_inds of the remaining BBs after the first batch selection
        }
        if self.parser.model in ["gcn", "gcn_delta"]:
            self._mkdir(
                f"saved_results/{self.dataset}/{self.parser.target_core}/gcn_params"
            )
            if os.path.exists(
                f"saved_results/{self.dataset}/{self.parser.target_core}/gcn_params/{self.result_filename.split('/')[-1]}"
            ):
                self.gcn_params = joblib.load(
                    f"saved_results/{self.dataset}/{self.parser.target_core}/gcn_params/{self.result_filename.split('/')[-1]}"
                )
                if len(self.gcn_params) != self.parser.n_bootstrap:
                    raise IndexError(
                        "The number of GCN hyperparameters does not match number of bootstraps. Check configuration."
                    )
            else:
                self.gcn_params = []
            gcn_params_at_start = deepcopy(self.gcn_params)

        bb_ind_list = np.arange(self.dataset.yield_array.shape[1])
        for i in tqdm(range(self.parser.n_bootstrap)):
            np.random.seed(42 + i)

            # Randomly selecting a few BBs out as the test set, as specified through the n_test keyword
            test_bb_inds = np.sort(
                np.random.choice(
                    bb_ind_list,
                    self.parser.n_test,
                    False,  # Not choosing the same building block multiple times
                )
            )
            train_bb_inds = [
                x for x in bb_ind_list if x not in test_bb_inds
            ]  # Actual indices in full set of BBs
            if self.parser.model in ["rfc_combined", "rfc_target"]:
                (
                    all_proba,
                    all_remaining_inds,
                    all_num_desired_in_first_batch,
                ) = self.evaluate_RF(
                    i,
                    train_bb_inds,
                    test_bb_inds,
                    source_array_tuple,
                    target_array_tuple,
                )
            elif self.parser.model == "rmap":
                (
                    all_proba,
                    all_remaining_inds,  
                    all_num_desired_in_first_batch,
                ) = self.evaluate_LP(train_bb_inds, test_bb_inds)
            elif self.parser.model == "knn":
                (
                    all_proba,
                    all_remaining_inds,
                    all_num_desired_in_first_batch,
                ) = self.evaluate_knn(train_bb_inds, test_bb_inds)
            elif self.parser.model == "svm":
                (
                    all_proba,
                    all_remaining_inds,
                    all_num_desired_in_first_batch,
                ) = self.evaluate_svm(train_bb_inds, test_bb_inds)
            elif self.parser.model == "gcn":
                (
                    all_proba,
                    all_remaining_inds,
                    all_num_desired_in_first_batch,
                ) = self.evaluate_gcn(
                    i,
                    train_bb_inds,
                    test_bb_inds,
                    source_array_tuple,
                    target_array_tuple,
                    gcn_params_at_start,
                )
            elif self.parser.model == "baseline":
                (
                    all_proba,
                    all_remaining_inds,
                    all_num_desired_in_first_batch,
                ) = self.evaluate_baseline(
                    train_bb_inds,
                    test_bb_inds,
                    source_array_tuple,
                    target_array_tuple,
                )

            for j, (proba, remaining_inds, desired) in enumerate(
                zip(all_proba, all_remaining_inds, all_num_desired_in_first_batch)
            ):
                results["bootstrap_id"].append(i)
                results["selection_id"].append(j)
                results["proba"].append(proba)
                y_bin = self._binarize_reactivity_classes(
                    target_array_tuple[1][train_bb_inds][remaining_inds]
                )
                results["target"].append(y_bin)
                results["train_bb_inds"].append(train_bb_inds)
                results["test_bb_inds"].append(test_bb_inds)
                results["Number of desired reactivity class in first batch"].append(
                    desired
                )
                results["remaining_inds"].append(remaining_inds)

        joblib.dump(results, self.result_filename)
        return results

    def _binarize_reactivity_classes(self, y, inds_to_not_touch=None):
        """Although the original evaluation of reactivity is multi-class (0, 1, 2) for evaluation purposes
         we tranform it to binary. For highly reactive target cores, we treat highest yielding class-2 as positives only
          while for slightly lower ones class-1 is also included as positives.

        Parameters
        ----------
        y : np.ndarray of shape (n_train_bbs,)
            Original reactivity array with multiple classes.
        inds_to_not_touch : list of ints
            When dealing with arrays that go in to training, there are indices that should not be touched at all.
            For example, when simulating missing data.

        Returns
        -------
        y_copy : np.ndarray of shape (n_train_bbs,)
            Binarized reactivity array depending on the target core.
        """
        y_copy = deepcopy(y)
        if inds_to_not_touch is not None:
            vals_to_not_touch = [x for i, x in enumerate(y) if i in inds_to_not_touch]
        if self.parser.target_core in [
            0,
            1,
            3,
            7,
            8,
        ]:  # For high reactivity cores, we want to focus on class-2
            y_copy[y_copy < 2] = 0
            y_copy[y_copy == 2] = 1
        else:  # For lower reactivity cores, even yields between 20~40% is a success
            y_copy[y_copy >= 1] = 1
        if inds_to_not_touch is not None:
            for i, val in zip(inds_to_not_touch, vals_to_not_touch):
                y_copy[i] = val
        return y_copy

    def select_first_batch(self, train_bb_inds, test_bb_inds, **kwargs):
        """Selecting the first batch of building blocks in the training set to evaluate against the target core of interest.

        Parameters
        ----------
        train_bb_inds : list of ints
            Original indices from all building blocks of those in the training dataset.
        test_bb_inds : list of ints
            Original indices from all building blocks of those selected as the test building blocks.

        Returns
        -------
        selected_inds : list of ints
            Indices of building blocks in the training dataset selected as useful ones to test out against the target core.
        """
        if self.parser.n1_strategy in ["rmap", "modularity"]:
            rmap = ReactivityMap(
                source_core_inds=None,
                target_core_ind=self.target_core_ind,
                test_bb_inds=test_bb_inds,
                dataset=self.dataset,
                mask=None,
            )
            if self.parser.n1_strategy == "rmap":
                selected_inds = [
                    rmap.select_first_batch(
                        self.parser.n1,
                        similarity_threshold=0.82,  # self.parser.threshold
                    )
                ]  # Indices within the train_bb_inds
                assert len(selected_inds) == 1
            else:  # Community detection through modularity maximization
                sim_mat = rmap.get_similarity_matrix()
                adj_mat = rmap._similarity_matrix_to_adjacency_matrix(
                    sim_mat, 0.82
                )  # self.parser.threshold)
                graph = nx.from_numpy_array(adj_mat)
                modularity_communities = nx.community.greedy_modularity_communities(
                    graph, weight="weight", best_n=self.parser.n1, cutoff=self.parser.n1
                )
                selected_inds = []
                for community in modularity_communities:
                    sub_adj_mat = adj_mat[list(community), :][:, list(community)]
                    degrees = np.sum(sub_adj_mat, axis=0)
                    highest_degree_node = np.argmax(degrees)
                    selected_inds.append(list(community)[highest_degree_node])
                selected_inds = [selected_inds]

        elif self.parser.n1_strategy == "uncertainty":
            model, X_train, y_train = (
                kwargs["model"],
                kwargs["X_train"],
                kwargs["y_train"],
            )
            X_test = kwargs["X_test"]
            model.fit(X_train, y_train)
            # Selecting most uncertain class-2's
            # We measure the entropy of each tree's predicted probabilities and average them.
            selected_inds = [
                np.argsort(entropy(model.predict_proba(X_test), axis=1))[
                    -1 * self.parser.n1 :
                ]
            ]

        elif self.parser.n1_strategy == "desc_cluster":
            X_train = kwargs["X_train"]
            kmeans_dist = KMeans(n_clusters=6, random_state=42).fit_transform(X_train)
            selected_inds = [[np.argmin(dist_row) for dist_row in kmeans_dist.T]]

        elif self.parser.n1_strategy == "random":
            selected_inds = [
                np.random.choice(np.arange(len(train_bb_inds)), self.parser.n1, False)
                for _ in range(self.parser.n1_random)
            ]
        return selected_inds

    @staticmethod
    def _take_care_of_single_class_proba(
        model, X, indices_of_interest, whether_X_equals_inds_of_interest=False
    ):
        """If training labels were of a single class, shape of predicted proba becomes 1-dimensional
        and this can throw an error.

        Parameters
        ----------
        model : ML predictor
            ML model that gives predicted probability values.
        X : np.ndarray of shape (n_instances, n_features)
            Input for the model.
        indices_of_interest : list of ints
            Indices of BBs or reactions that we care for
        whether_X_equals_inds_of_interest : bool
            Whether X includes input features of other BBs / reactions.
            Needed to take care of LP which includes all BBs.

        Returns
        -------
        pred_proba : np.ndarray of shape (n_indices_of_interest,)
            Predicted probability values. If the model was trained on only 0's it will return 0s and only 1's will be 1 only.
        """
        if model.predict_proba(X).shape[1] == 2:
            if not whether_X_equals_inds_of_interest:
                pred_proba = model.predict_proba(X)[indices_of_interest, 1]
            else:
                pred_proba = model.predict_proba(X)[:, 1]
        else:
            if np.sum(model.predict(X)) == 0:
                pred_proba = np.zeros(len(indices_of_interest))
            elif np.sum(model.predict(X)) == X.shape[0]:
                pred_proba = np.ones(len(indices_of_interest))
        return pred_proba

    def evaluate_LP(self, train_bb_inds, test_bb_inds):
        """Makes predictions with label propagation on the specified evaluation setup.

        Parameters
        ----------
        train_bb_inds : list of ints
            Indices of building blocks selected as training data.
        test_bb_inds : list of ints
            Indices of building blocks left out as test data.

        Returns
        -------
        all_rmap_proba: list of np.ndarrays of shape (n_remaining_bb_inds, )
            Predicted probability values of desired reactivity class across all selection folds.
        all_remaining_inds : list of list of ints
            Indices of training building blocks that remain after selecting the first batch of bbs.
        """
        if self.parser.subgraph:
            source_cores = [x for x in [0, 1, 3, 7, 8] if x != self.target_core_ind]
        else:
            source_cores = None
        rmap = ReactivityMap(
            source_cores, self.target_core_ind, test_bb_inds, self.dataset, mask=None
        )
        _ = rmap.get_similarity_matrix(tanimoto=self.parser.tanimoto)
        all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
        all_rmap_proba = []
        all_remaining_inds = []
        all_num_desired_in_first_batch = []
        for rmap_selected_inds in all_selected_inds:
            rmap_remaining_inds = [
                x for x in range(len(train_bb_inds)) if (x not in rmap_selected_inds)
            ]  # Indices within the train_bb_inds
            all_num_desired_in_first_batch.append(
                sum(
                    rmap.target_reactivity_array[rmap_selected_inds]
                    >= self.desired_class
                )
            )
            if self.parser.tanimoto :
                threshold=0.33
            else :
                threshold=0.82
            adj = rmap._similarity_matrix_to_adjacency_matrix(
                rmap.get_similarity_matrix(tanimoto=self.parser.tanimoto), threshold
            )  # self.parser.threshold)
            adj += 0.001  # To prevent division error

            def _kernel(X, y):
                return adj

            X_ohe = np.identity(len(train_bb_inds))
            y = -1 * np.ones(len(train_bb_inds))
            y[rmap_selected_inds] = rmap.target_reactivity_array[
                rmap_selected_inds
            ]  # Simulating we obtained results of the first batch BBs with the new core

            all_remaining_inds.append(rmap_remaining_inds)
            y = self._binarize_reactivity_classes(
                y, inds_to_not_touch=list(np.where(y == -1)[0])
            )
            lp = LabelPropagation(kernel=_kernel, n_jobs=-1)
            lp.fit(X_ohe, y)
            raw_pred_proba = lp.predict_proba(X_ohe)
            lp_pred_proba = raw_pred_proba[:, -1]
            all_rmap_proba.append(lp_pred_proba[rmap_remaining_inds])
        return all_rmap_proba, all_remaining_inds, all_num_desired_in_first_batch

    def evaluate_knn(self, train_bb_inds, test_bb_inds):
        """Makes predictions with kNN using the reactivity similarity matrix.

        Parameters
        ----------
        train_bb_inds : list of ints
            Indices of building blocks selected as training data.
        test_bb_inds : list of ints
            Indices of building blocks left out as test data.

        Returns
        -------
        all_knn_proba: list of np.ndarrays of shape (n_remaining_bb_inds, )
            Predicted probability values of desired reactivity class across all selection folds.
        all_remaining_inds : list of list of ints
            Indices of training building blocks that remain after selecting the first batch of bbs.
        """
        if self.parser.subgraph:
            source_cores = [x for x in [0, 1, 3, 7, 8] if x != self.target_core_ind]
        else:
            source_cores = None
        rmap = ReactivityMap(
            source_cores, self.target_core_ind, test_bb_inds, self.dataset, mask=None
        )
        dist_mat = 1-rmap.get_similarity_matrix()
        all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
        all_knn_proba = []
        all_remaining_inds = []
        all_num_desired_in_first_batch = []
        for rmap_selected_inds in all_selected_inds:
            rmap_remaining_inds = [
                x for x in range(len(train_bb_inds)) if (x not in rmap_selected_inds)
            ]  # Indices within the train_bb_inds
            all_remaining_inds.append(rmap_remaining_inds)
            all_num_desired_in_first_batch.append(
                sum(
                    rmap.target_reactivity_array[rmap_selected_inds]
                    >= self.desired_class
                )
            )
            X_train = dist_mat[np.ix_(rmap_selected_inds, rmap_selected_inds)]
            y_train = self._binarize_reactivity_classes(
                rmap.target_reactivity_array[
                    rmap_selected_inds
                ]
            )  # Simulating we obtained results of the first batch BBs with the new core
            if np.sum(y_train) not in [0, len(y_train)]:
                neigh = GridSearchCV(
                    KNeighborsClassifier(weights="distance", metric="precomputed"),
                    param_grid={
                        "n_neighbors": [1, 2, 3],
                    },
                    scoring="accuracy",
                    n_jobs=-1,
                    cv=LeaveOneOut(),
                )
                neigh.fit(X_train, y_train)
                X_remaining = dist_mat[np.ix_(rmap_remaining_inds, rmap_selected_inds)]
                raw_pred_proba = neigh.predict_proba(X_remaining)
                knn_pred_proba = raw_pred_proba[:, -1]
            else :
                knn_pred_proba = np.zeros(len(rmap_remaining_inds)) if np.sum(y_train) == 0 else np.ones(len(rmap_remaining_inds))
            all_knn_proba.append(knn_pred_proba)
        return all_knn_proba, all_remaining_inds, all_num_desired_in_first_batch

    def evaluate_svm(self, train_bb_inds, test_bb_inds):
        """Makes predictions with SVM using the reactivity similarity matrix.

        Parameters
        ----------
        train_bb_inds : list of ints
            Indices of building blocks selected as training data.
        test_bb_inds : list of ints
            Indices of building blocks left out as test data.

        Returns
        -------
        all_svm_proba: list of np.ndarrays of shape (n_remaining_bb_inds, )
            Predicted probability values of desired reactivity class across all selection folds.
        all_remaining_inds : list of list of ints
            Indices of training building blocks that remain after selecting the first batch of bbs.
        """
        if self.parser.subgraph:
            source_cores = [x for x in [0, 1, 3, 7, 8] if x != self.target_core_ind]
        else:
            source_cores = None
        rmap = ReactivityMap(
            source_cores, self.target_core_ind, test_bb_inds, self.dataset, mask=None
        )
        _ = rmap.get_similarity_matrix(tanimoto=self.parser.tanimoto)
        all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
        all_svm_proba = []
        all_remaining_inds = []
        all_num_desired_in_first_batch = []
        for rmap_selected_inds in all_selected_inds:
            rmap_remaining_inds = [
                x for x in range(len(train_bb_inds)) if (x not in rmap_selected_inds)
            ]  # Indices within the train_bb_inds
            all_remaining_inds.append(rmap_remaining_inds)
            all_num_desired_in_first_batch.append(
                sum(
                    rmap.target_reactivity_array[rmap_selected_inds]
                    >= self.desired_class
                )
            )
            adj = rmap._similarity_matrix_to_adjacency_matrix(
                rmap.get_similarity_matrix(), 0.82
            )  # self.parser.threshold)
            adj += 0.001  # To prevent division error
            X_train = adj[np.ix_(rmap_selected_inds, rmap_selected_inds)]
            y_train = self._binarize_reactivity_classes(
                rmap.target_reactivity_array[
                    rmap_selected_inds
                ]
            )  # Simulating we obtained results of the first batch BBs with the new core
            if np.sum(y_train) not in [0, len(y_train)]:
                svm = SVC(kernel="precomputed", probability=True, random_state=42)
                svm.fit(X_train, y_train)
                X_test = adj[np.ix_(rmap_remaining_inds, rmap_selected_inds)]
                raw_pred_proba = svm.predict_proba(X_test)
                svm_pred_proba = raw_pred_proba[:, -1]
            else :
                svm_pred_proba = np.zeros(len(rmap_remaining_inds)) if np.sum(y_train) == 0 else np.ones(len(rmap_remaining_inds))
            all_svm_proba.append(svm_pred_proba)
        return all_svm_proba, all_remaining_inds, all_num_desired_in_first_batch

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _compute_reactivity_similarity(
        self, array1, array2, weights="uniform", zero_same_factor=1
    ):
        """Computes the reactivity similarity between arrays of yield classes.
        If two values are the same --> assign score of 2
        If two values are both positive with different values --> take reciprocal of differences
        If two values are different with one being zero --> 0.

        Parameters
        ---------
        array1, array2 : np.ndarray of shape (n_boronics,)
            Yield class for each halide, where each cell corresponds to a boronic.
        weights : str 'uniform' or np.ndarray of shape (n_boronics)
            Which weight scheme to use.
        zero_same_factor : float
            The weight of two values both being zero. The score will be multiplied by this value.

        Returns
        -------
        score : float
            Similarity score.
        """
        if type(weights) == str and weights == "uniform":
            weights = np.ones(len(array1))
        score = 0
        for elem1, elem2, weight in zip(array1, array2, weights):
            if elem1 == elem2:
                if elem1 != 0:
                    score += 2 * weight
                else:
                    score += 2 * zero_same_factor * weight
            elif elem1 * elem2 == 0:
                score += 0
            else:
                score += weight / abs(elem1 - elem2)
        score /= 2 * np.sum(weights)
        if type(score) == np.ndarray:
            score = score[0]
        return score

    def evaluate_gcn(
        self,
        bootstrap_ind,
        train_bb_inds,
        test_bb_inds,
        source_array_tuple,
        target_array_tuple,
        gcn_params,
    ):
        """First obtains results of the first batch of experiments as specified by the parser.
        Then, a graph convolutional network is trained to predict the reactivity classes of remaining building blocks with the target core.
        """
        # First get the first batch of results
        rmap = ReactivityMap(
            None, self.target_core_ind, test_bb_inds, self.dataset, mask=None
        )
        similarity_matrix = (
            rmap.get_similarity_matrix()
        )  # TODO: need to update when there are missing values in training data
        all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
        adj_matrix = rmap._similarity_matrix_to_adjacency_matrix(
            similarity_matrix, 0.82  # self.parser.threshold
        )

        scaler = StandardScaler()
        if self.parser.feature in ["desc", "fp"]:
            raw_node_features = target_array_tuple[0][train_bb_inds][
                :, self.dataset.core_descriptors.shape[1] :
            ]
            node_features_std = scaler.fit_transform(raw_node_features)
        elif self.parser.feature == "ohe":
            node_features_std = np.eye(len(train_bb_inds))

        all_gcn_proba = []
        all_remaining_inds = []
        all_num_desired_in_first_batch = (
            []
        )  # To keep track of how many desired results are identified from the first batch of experiments.
        each_gcn_params = []
        for a, selected_inds in enumerate(all_selected_inds):
            gcn_remaining_inds = [
                x for x in range(len(train_bb_inds)) if x not in selected_inds
            ]
            all_num_desired_in_first_batch.append(
                sum(rmap.target_reactivity_array[selected_inds] >= self.desired_class)
            )
            # Defining pyg data object with the target core reactivity labels
            y_to_use = self._binarize_reactivity_classes(
                target_array_tuple[1][train_bb_inds].flatten()
            )
            train_mask = index_to_mask(
                torch.tensor(selected_inds), size=len(train_bb_inds)
            )
            test_mask = ~train_mask
            edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()
            row, col = edge_index
            edge_weight = adj_matrix[row, col]
            data = Data(
                x=torch.tensor(node_features_std, dtype=torch.float),
                edge_index=torch.tensor(adj_matrix).nonzero().t().contiguous(),
                edge_weight=torch.tensor(edge_weight, dtype=torch.float),
                y=torch.tensor(y_to_use, dtype=torch.float),
                train_mask=train_mask,
                test_mask=test_mask,
            )
            data.to(self.device)
            # Hyperparameter search if not previously done.
            if gcn_params == []:

                def optuna_objective(trial):
                    if self.parser.feature in ["desc", "ohe"]:
                        hidden_size1_trial = trial.suggest_int(
                            "hidden_size1", 10, 40, step=10
                        )
                        hidden_size2_trial = trial.suggest_int(
                            "hidden_size2", 0, 10, step=5
                        )
                    elif self.parser.feature == "fp":
                        hidden_size1_trial = (
                            trial.suggest_int("hidden_size1", 20, 100, step=20),
                        )
                        hidden_size2_trial = (
                            trial.suggest_int("hidden_size2", 5, 20, step=5),
                        )
                    model = GCN(
                        n_features=data.x.shape[1],
                        hidden_size1=hidden_size1_trial,
                        hidden_size2=hidden_size2_trial,
                    )
                    model.to(self.device)
                    opt = torch.optim.Adam(
                        model.parameters(),
                        lr=trial.suggest_float("learning_rate", 0.0001, 0.01, log=True),
                        # weight_decay=trial.suggest_float("weight_decay", 0, 0.01, step=0.005)
                        weight_decay=0,
                    )
                    for epoch in range(300):
                        model.train()
                        opt.zero_grad()
                        out = model(data.x, data.edge_index, data.edge_weight)
                        loss = nn.BCEWithLogitsLoss()(
                            out[data.train_mask], data.y[data.train_mask].reshape(-1, 1)
                        )
                        loss.backward()
                        opt.step()

                        trial.report(loss.cpu(), epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                        else:
                            if epoch % 20 == 0:
                                print("EPOCH", epoch)
                                print("TRAINING LOSS", loss.cpu().item())
                    return loss

                study = optuna.create_study(
                    pruner=optuna.pruners.MedianPruner(),
                    direction="minimize",
                    sampler=None,
                )
                study.optimize(optuna_objective, n_trials=20)
                best_trial = study.best_trial
                best_params_dict = best_trial.params
                # self.gcn_params.append(best_params_dict)
                each_gcn_params.append(best_params_dict)

            else:
                print("Using hyperparameters of previously trained models...")
                best_params_dict = gcn_params[bootstrap_ind][a]
            # Training model with best hyperparameters
            if "hidden_size2" in best_params_dict.keys():
                opt_hs2 = best_params_dict["hidden_size2"]
            else:
                opt_hs2 = 0
            opt_model = GCN(
                n_features=data.x.shape[1],
                hidden_size1=best_params_dict["hidden_size1"],
                hidden_size2=opt_hs2,
            ).to(self.device)
            opt = torch.optim.Adam(
                opt_model.parameters(),
                lr=best_params_dict["learning_rate"],
                # weight_decay=best_params_dict["weight_decay"]
                weight_decay=0,
            )

            def train():
                opt_model.train()
                opt.zero_grad()
                out = opt_model(data.x, data.edge_index, data.edge_weight)
                loss = nn.BCEWithLogitsLoss()(
                    out[data.train_mask], data.y[data.train_mask].reshape(-1, 1)
                )
                loss.backward()
                opt.step()
                return loss

            print("=======================")
            print("Training final model...")
            print("=======================")
            for epoch in range(300):
                loss = train()
                if epoch % 20 == 0:
                    print(f"    Epoch: {epoch:03d}, Loss: {loss.cpu().item():.4f}")

            def predict():
                opt_model.eval()
                out = opt_model(data.x, data.edge_index, data.edge_weight)
                return out

            out = predict().detach().cpu().numpy()
            gcn_proba = self.sigmoid(out).flatten()
            all_gcn_proba.append(gcn_proba[gcn_remaining_inds])
            all_remaining_inds.append(gcn_remaining_inds)
            self.gcn_params.append(each_gcn_params)
        return all_gcn_proba, all_remaining_inds, all_num_desired_in_first_batch

    def evaluate_RF(
        self,
        bootstrap_ind,
        train_bb_inds,
        test_bb_inds,
        source_array_tuple,
        target_array_tuple,
    ):
        gcv = GridSearchCV(
            RandomForestClassifier(random_state=42 + bootstrap_ind),
            param_grid={
                "n_estimators": [10, 20, 50],
                "max_depth": [2, 3, 5, None],
            },
            scoring="accuracy",
            n_jobs=-1,
            cv=5,
        )
        X_train = self._unroll_arrays_for_rf(
            None, source_array_tuple[0][:, train_bb_inds, :]
        )
        y_train = self._unroll_arrays_for_rf(
            None, source_array_tuple[1][:, train_bb_inds]
        )
        rmap = ReactivityMap(
            None, self.target_core_ind, test_bb_inds, self.dataset, mask=None
        )
        if self.parser.n1_strategy == "uncertainty":
            all_selected_inds = self.select_first_batch(
                train_bb_inds,
                test_bb_inds,
                model=gcv,
                X_train=X_train,
                y_train=y_train,
                X_test=target_array_tuple[0][train_bb_inds],
            )
        elif self.parser.n1_strategy == "random":
            all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
        elif self.parser.n1_strategy == "desc_cluster":
            X_bb_train = target_array_tuple[0][train_bb_inds][
                :, self.dataset.core_descriptors.shape[1] :
            ]
            all_selected_inds = self.select_first_batch(
                train_bb_inds, test_bb_inds, X_train=X_bb_train
            )
        elif self.parser.n1_strategy == "rmap":
            _ = rmap.get_similarity_matrix()
            all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
        all_rfc_proba = []
        all_remaining_inds = []
        all_num_desired_in_first_batch = []
        for rfc_selected_inds in all_selected_inds:
            rfc_remaining_inds = [
                x for x in range(len(train_bb_inds)) if x not in rfc_selected_inds
            ]

            X_target = target_array_tuple[0][train_bb_inds][rfc_selected_inds]
            y_target = target_array_tuple[1][train_bb_inds][rfc_selected_inds]

            all_remaining_inds.append(rfc_remaining_inds)
            all_num_desired_in_first_batch.append(
                np.sum(y_target >= self.desired_class)
            )
            X_target_remaining = target_array_tuple[0][train_bb_inds][
                rfc_remaining_inds
            ]
            y_train = self._binarize_reactivity_classes(y_train)
            y_target = self._binarize_reactivity_classes(y_target)

            gcv.fit(
                np.vstack((X_train, X_target)),
                np.concatenate((y_train, y_target)),
                sample_weight=[1] * len(y_train)
                + [3] * len(y_target),  # Need to emphasize target samples
            )
            if self.parser.model == "rfc_target":
                # Retraining only on the target core data using the best estimator found for combined data
                gcv = deepcopy(gcv.best_estimator_)
                gcv.fit(X_target[:, self.dataset.core_descriptors.shape[1] :], y_target)
                X_target_to_use = X_target_remaining[
                    :, self.dataset.core_descriptors.shape[1] :
                ]
            else:
                X_target_to_use = X_target_remaining
            proba = gcv.predict_proba(X_target_to_use)
            if proba.shape[1] > 1:
                all_rfc_proba.append(gcv.predict_proba(X_target_to_use)[:, 1])
            else:
                predictions = gcv.predict(X_target_to_use)
                all_rfc_proba.append(
                    np.array([np.unique(predictions) for _ in range(len(predictions))])
                )
        return all_rfc_proba, all_remaining_inds, all_num_desired_in_first_batch

    def evaluate_baseline(
        self, train_bb_inds, test_bb_inds, source_array_tuple, target_array_tuple
    ):
        """Evaluating the baseline where predicted probabilities are computed as portions of cores in the source dataset that were in each reactivity class.
        Also, we assume that the first batch is greedily selected with highest averaging yields.
        """
        if self.parser.n1_strategy == "rmap":
            rmap = ReactivityMap(
                None, self.target_core_ind, test_bb_inds, self.dataset, mask=None
            )
            _ = rmap.get_similarity_matrix()
            all_selected_inds = self.select_first_batch(train_bb_inds, test_bb_inds)
            # print(all_selected_inds)
        # Getting the highest average-yielding building blocks
        elif self.parser.n1_strategy == "greedy":
            training_yield_array = source_array_tuple[2][:, train_bb_inds]
            all_selected_inds = [
                np.argsort(np.mean(training_yield_array, axis=0))[-1 * self.parser.n1 :]
            ]
        all_remaining_inds = [
            [x for x in np.arange(len(train_bb_inds)) if x not in all_selected_inds[0]]
        ]
        # Then getting probabilities for remaining indices by taking the portion in the source cores that gave desired classes
        all_baseline_proba = []
        for column_ind in all_remaining_inds[0]:
            column = source_array_tuple[1][:, train_bb_inds[column_ind]]
            all_baseline_proba.append(
                np.sum(column == self.desired_class) / len(column)
            )
        all_baseline_proba = [np.array(all_baseline_proba)]

        # Number of desired class in greedy first batch selection
        all_num_desired_in_first_batch = [
            np.sum(target_array_tuple[1][all_selected_inds] == self.desired_class)
        ]

        return all_baseline_proba, all_remaining_inds, all_num_desired_in_first_batch

    @staticmethod
    def _unroll_arrays_for_rf(mask, array):
        """Transforms the output array such that it is compatible with random forest classifiers.

        Parameters
        ----------
        mask : np.ndarray
            Values with 1 simulate data that we have not obtained in the training dataset.
        array : np.ndarray
            Input or output values to rearrange for use with conventional models like RF.

        Returns
        -------
        unrolled_array : np.ndarray
            Transformed array.
        """
        if mask is not None:
            if array.ndim == 2:
                return array[mask != 1].flatten()
            elif array.ndim == 3:
                indiv_arrays = []
                for i, j in zip(np.where(mask != 1)[0], np.where(mask != 1)[1]):
                    indiv_arrays.append(array[i, j, :])
                return np.vstack(tuple(indiv_arrays))
        else:
            if array.ndim == 2:
                return array.flatten()
            elif array.ndim == 3:
                indiv_arrays = []
                for i in range(array.shape[0]):
                    for j in range(array.shape[1]):
                        indiv_arrays.append(array[i, j, :])
                return np.vstack(tuple(indiv_arrays))


def run_evaluation(parser):
    Evaluator(parser).evaluate()


if __name__ == "__main__":
    parser = parse_args()
    run_evaluation(parser)
