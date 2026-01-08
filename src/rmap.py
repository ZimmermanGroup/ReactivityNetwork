from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
import os
from dataset import *
from math import log, e


class ReactivityMap:
    """Object for modeling with the reactivity map.
    • first defines previously collected data and building blocks that haven't been used at all (as defined by the test_bb_inds parameter).
    • generates excel files to use with Gephi for visualization.
    • selects the first and second sets of boronics to screen based on their clustering with reactivity.

    Parameters
    ----------
    source_core_inds : list of ints or None
        Indexes of dataset.core_smiles that is simulated to have been used before.
        if None : assume all other than the target is in the training data.
    target_core_ind : int
        Index of dataset.core_smiles that is of interest and want to predict reactivity of.
    test_bb_inds : list of ints
        Building block indices to set aside as those not seen before.
    dataset : Dataset object
        Dataset object from dataset.py
    mask : np.ndarray of shape (n_source_cores, n_train_bb)
        Whether each substrate pair in the training dataset is masked
    """

    def __init__(
        self, source_core_inds, target_core_ind, test_bb_inds, dataset, mask=None
    ):
        self.target_core_ind = target_core_ind
        if source_core_inds is not None:
            self.source_core_inds = source_core_inds
        else:
            self.source_core_inds = [
                i for i in range(len(dataset.core_smiles)) if i != target_core_ind
            ]
        self.test_bb_inds = test_bb_inds
        self.train_bb_inds = [
            i for i in range(len(dataset.bb_smiles)) if i not in self.test_bb_inds
        ]
        self.dataset = dataset
        self.mask = mask

        self._initialize_arrays()
        if self.mask is not None:
            self._apply_mask_to_arrays()

    def _initialize_arrays(self):
        self.source_reactivity_array = self.dataset.reactivity_array[
            self.source_core_inds
        ][:, self.train_bb_inds]
        self.source_yield_array = self.dataset.yield_array[self.source_core_inds][
            :, self.train_bb_inds
        ]
        self.target_reactivity_array = self.dataset.reactivity_array[
            self.target_core_ind
        ][self.train_bb_inds]

    def _apply_mask_to_arrays(self):
        self.source_reactivity_array = np.array(
            self.source_reactivity_array, dtype=np.float16
        )
        self.source_reactivity_array[self.mask == 1] = np.nan
        self.source_yield_array[self.mask == 1] = np.nan

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

    def get_similarity_matrix(self, imputation=None, gephi_filename=None):
        """Prepares an adjacency matrix in terms of reactivity similarity scores.
        Saves excel files to be processed in gephi for further visualization, if desired.
        Since the similarity threshold can be controlled in gephi, we simply get all reactivity similarity values between boronics.

        Parameters
        ----------
        gephi_filename : str or None
            Name of excel file to save for downstream visualization with gephi.
        imputation : str {bb, core, matrix_completion} or None
            If there are empty datapoints (i.e., self.mask is not None), how to fill in the values to compute reactivity similarity.
            • bb : filling the missing value with average value of reactivity classes IN THE SAME BB
            • core : filling the missing value with average value of reactivity classes IN THE SAME CORE
            • similarity : find the most similar boronic using the outcomes from cores that overlap
            • rfc : fill in the empty ones using a random forest classifier.

        Returns
        -------
        similarity_matrix : np.ndarray of shape (n_BBs, n_BBs)
            Raw reaction similarity values for all pairs of BBs.
        """
        # Initializing node and edge dictionaries
        nodes = {"Id": [], "Smiles": []}
        for i in self.source_core_inds:
            nodes.update({f"Core{i+1}": []})
        if type(self.dataset == SuzukiDataset):
            nodes.update({"Centroid": [], "Cluster": []})
        edges = {"Source": [], "Target": [], "Type": [], "Weight": []}

        if len(self.test_bb_inds) == 0:
            train_bb_smiles = self.dataset.bb_smiles
        else:
            train_bb_smiles = [self.dataset.bb_smiles[x] for x in self.train_bb_inds]

        imputed_array = self._impute_missing_values(imputation, train_bb_smiles)
        similarity_matrix = self._calculate_similarity(
            imputed_array, nodes, edges, gephi_filename, train_bb_smiles
        )

        self.similarity_matrix = similarity_matrix
        self.imputed_reactivity_array = imputed_array
        return self.similarity_matrix

    def _impute_missing_values(self, imputation, train_bb_smiles):
        """Fills in the missing data in the training dataset with the defined strategy.

        Parameters
        ----------
        imputation : str
            How to fill in the missing values of the source dataset.
        train_bb_smiles : list of str
            List of SMILES strings of the boronic building blocks in the training dataset.

        Returns
        -------
        imputed_array : np.ndarray of shape (n_source_cores, n_train_bbs)
            Imputed reaction outcome array.
        """
        if self.mask is not None:
            imputed_array = deepcopy(self.source_reactivity_array)
            # Specific imputation strategies
            if imputation in ["bb", "core"]:
                self._impute_by_mean(imputed_array, imputation)
            elif imputation == "similarity":
                self._impute_by_similarity(imputed_array, train_bb_smiles)
            elif imputation == "rfc":
                self._impute_by_rfc(imputed_array)
        else:
            imputed_array = self.source_reactivity_array
        return imputed_array

    def _impute_by_mean(self, imputation, imputed_array):
        """Imputes missing data by taking the average across the selected axis as defined by 'imputation'.

        Parameters
        ----------
        imputation : str {'bb', 'core'}
            Axis (boronic building block or core) to take the average across.
        imputed_array : np.ndarray of shape (n_source_cores, n_train_bbs)
            The array to fill in the gap of.
        """
        axis = 0 if imputation == "bb" else 1
        each_core_mean = np.nanmean(imputed_array, axis=axis)
        inds_to_recover = np.where(np.isnan(imputed_array))
        imputed_array[inds_to_recover] = np.take(
            each_core_mean, inds_to_recover[1 - axis]
        )

    def _impute_by_similarity(self, imputed_array, train_bb_smiles):
        """Imputes missing data by taking the weighted average of reactivity classes from
        building blocks that have the most overlapping same value of reactivity classes.

        Parameters
        ----------
        imputed_array : np.ndarray of shape (n_source_cores, n_train_bbs)
            The array to fill in the gap of.
        train_bb_smiles : list of str
            List of SMILES strings of the boronic building blocks in the training dataset.
        """
        for i, smiles in enumerate(train_bb_smiles):
            unmasked_portion_of_current_bb = ~np.isnan(imputed_array[:, i].flatten())
            unmasked_portion_of_all_bbs = ~np.isnan(imputed_array.T)
            number_of_core_intersections = np.sum(
                unmasked_portion_of_current_bb & unmasked_portion_of_all_bbs,
                axis=1,
            )
            num_bb_to_consider = int(0.7 * len(train_bb_smiles))
            ### Narrows down to building blocks that share the most number of cores
            inds_with_most_intersections = np.argpartition(
                number_of_core_intersections, num_bb_to_consider
            )[num_bb_to_consider:]
            similarities = []
            similarity_inds = []
            for x in inds_with_most_intersections:
                if x != i:
                    similarities.append(
                        self._compute_reactivity_similarity(
                            imputed_array[:, i].flatten()[
                                unmasked_portion_of_current_bb
                                & unmasked_portion_of_all_bbs[x].flatten()
                            ],
                            imputed_array[:, x].flatten()[
                                unmasked_portion_of_current_bb
                                & unmasked_portion_of_all_bbs[x].flatten()
                            ],
                        )
                    )
                    similarity_inds.append(x)

            median_similarity = np.median(similarities)
            most_similar_inds = [
                ind
                for val, ind in zip(similarities, similarity_inds)
                if val >= median_similarity
            ]
            imputed_array[np.isnan(imputed_array[:, i]), i] = [
                round(x) if not np.isnan(x) else np.nan
                for x in np.nanmean(imputed_array[:, most_similar_inds], axis=1)[
                    np.isnan(imputed_array[:, i])
                ]
            ]
            if np.sum(np.isnan(imputed_array[:, i])) > 0:
                imputed_array[np.isnan(imputed_array[:, i])] = np.nanmean(
                    imputed_array[:, i]
                )

    def _impute_by_rfc(self, imputed_array):
        """Imputes missing data by using a multiclass random forest classifier.

        Parameters
        ----------
        imputed_array : np.ndarray of shape (n_source_cores, n_train_bbs)
            The array to fill in the gap of.
        """
        y_unmasked = self.dataset.reactivity_array
        X_fp_unmasked = self.dataset.X_fp
        y_masked = np.array(
            [
                y_unmasked[row_ind, col_ind]
                for row_ind, col_ind in zip(*np.where(self.mask == 0))
            ]
        )
        X_fp_masked = np.vstack(
            tuple(
                [
                    X_fp_unmasked[row_ind, col_ind]
                    for row_ind, col_ind in zip(*np.where(self.mask == 0))
                ]
            )
        )
        rfc_gcv = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid={
                "n_estimators": [10, 25, 50, 100],
                "max_depth": [2, 3, 5],
            },
            n_jobs=-1,
            scoring="accuracy",
            cv=5,
        )
        rfc_gcv.fit(X_fp_masked, y_masked)
        X_fp_remaining = np.vstack(
            tuple(
                [
                    X_fp_unmasked[row_ind, col_ind]
                    for row_ind, col_ind in zip(*np.where(self.mask == 1))
                ]
            )
        )
        y_pred = rfc_gcv.predict(X_fp_remaining)
        for num, (row_ind, col_ind) in enumerate(zip(*np.where(self.mask == 1))):
            assert np.isnan(imputed_array[row_ind, col_ind])
            imputed_array[row_ind, col_ind] = y_pred[num]

    def _calculate_similarity(
        self, imputed_array, nodes, edges, gephi_filename, train_bb_smiles
    ):
        """Computes reactivity similarity between all pairs of building blocks.

        Parameters
        ----------
        imputed_array : np.ndarray of shape (n_source_cores, n_training_bbs)
            Array of reactivity classes between cores and bbs without empty values.
        nodes : dict
            Dictionary with information of all building block nodes.
        edges : dict
            Dictionary with information of edges between all pairs of nodes.
        gephi_filename : str
            Path to the gephi excels.
        train_bb_smiles : list of str
            List of SMILES strings of the boronic building blocks in the training dataset.

        Returns
        -------
        similarity_matrix : np.ndarray of shape (n_train_bbs, n_train_bbs)
            All reactivity similarity values between all pairs of building blocks in the training dataset.
        """
        similarity_matrix = np.identity(len(train_bb_smiles))
        n_train_bb = len(train_bb_smiles)
        for i, smiles in enumerate(train_bb_smiles):
            weights = [
                self._compute_reactivity_similarity(
                    imputed_array[:, i], imputed_array[:, x]
                )
                for x in range(i + 1, n_train_bb)
            ]
            if gephi_filename is not None:
                self._update_nodes(nodes, i, smiles, imputed_array)
                self._update_edges(edges, i, weights, n_train_bb)

            similarity_matrix[i, i + 1 :] = weights
        similarity_matrix += similarity_matrix.T
        similarity_matrix -= np.identity(n_train_bb)
        if gephi_filename is not None:
            self._save_to_excel(nodes, edges, gephi_filename)
        return similarity_matrix

    def _update_nodes(self, nodes, index, smiles, imputed_array):
        """Updates the node-information dictionary which will be further used to visualize the reactivity map with Gephi.

        Parameters
        ----------
        nodes : dict
            Dictionary to update.
        index : int
            Index of the building block.
        smiles : str
            SMILES string of the building block.
        imputed_array : np.ndarray of shape (n_source_cores, n_training_bbs)
            Reaction outcome array without empty values.

        Returns
        -------
        None
        """
        nodes["Id"].append(self.train_bb_inds[index] + 1)
        nodes["Smiles"].append(smiles)
        boronics_descriptors = pd.read_csv(
            "../data/relay_suzuki/boronic_descriptors.csv",
            usecols=["reactant_2&smiles", "Centroid", "Cluster"],
        )
        for j, core_id in enumerate(self.source_core_inds):
            nodes[f"Core{core_id+1}"].append(imputed_array[j, index])

        centroid = boronics_descriptors[
            boronics_descriptors["reactant_2&smiles"] == smiles
        ]["Centroid"].values[0]
        if centroid > -1:
            centroid = 1
        nodes["Centroid"].append(centroid)
        nodes["Cluster"].append(
            boronics_descriptors[boronics_descriptors["reactant_2&smiles"] == smiles][
                "Cluster"
            ].values[0]
        )

    def _update_edges(self, edges, i, weights, n_train_bb):
        """Updates the edge-information dictionary which will be further used to visualize the reactivity map with Gephi.

        Parameters
        ----------
        edges : dict
            Dictionary to update.
        i : int
            Index of the building block of interest. All edges from this building block will be recorded.
        weights : list of floats
            Reactivity similarity values between the building block of interest and all others.
        n_train_bb : int
            Number of building blocks in the training dataset.

        Returns
        -------
        None
        """
        edges["Source"].extend([self.train_bb_inds[i] + 1] * (n_train_bb - i - 1))
        edges["Target"].extend([x + 1 for x in self.train_bb_inds[i + 1 :]])
        edges["Type"].extend(["Undirected"] * (n_train_bb - i - 1))
        edges["Weight"].extend(weights)

    def _save_to_excel(self, nodes, edges, gephi_filename):
        """Saves the node and edge information saved as dictionaries into an excel, which can be used for
        downstream visualization of the reactivity map with Gephi.

        Parameters
        ----------
        nodes, edges: dict
            Dictionaries with node and edge informations.
        gephi_filename : str
            Name to save the files.

        Returns
        -------
        similarity_matrix : np.ndarray of shape (n_train_bbs, n_train_bbs)
            Reactivity similarity values collected in a matrix.
        """
        nodes_df = pd.DataFrame(nodes)
        nodes_df["Reactivity"] = nodes_df.iloc[
            :, 2 : 2 + len(self.source_core_inds)
        ].sum(axis=1)
        edges_df = pd.DataFrame(edges)

        if gephi_filename is not None:
            if not os.path.exists("gephi_excels/"):
                os.mkdir("gephi_excels/")
            nodes_df.to_excel(
                f"gephi_excels/{self.dataset.__str__()}_nodes_"
                + gephi_filename
                + ".xlsx",
                index=False,
            )
            edges_df.to_excel(
                f"gephi_excels/{self.dataset.__str__()}_edges_"
                + gephi_filename
                + ".xlsx",
                index=False,
            )
        # return self.similarity_matrix

    def _similarity_matrix_to_adjacency_matrix(self, sim_mat, threshold):
        """The reactivity similarity values themselves are not fully useful to formulate a network of building blocks.
        So we apply a threshold that defines a similarity value over this can be meaningfully thought as similar in terms of reactivity.
        After applying this, some nodes can be left alone. We connect these to one node that has the highest similarity value.

        Parameters
        ----------
        sim_mat : np.ndarray of shape (n_train_bbs, n_train_bbs)
            Reactivity similarity matrix.
        threshold : float
            Value to determine the presence of an edge in the reactivity map.

        Returns
        -------
        adj_mat : np.ndarray of shape (n_train_bbs, n_train_bbs)
            Adjacency matrix to be used for treating the building blocks as a part of the reactivity network.
        """
        adj_mat = deepcopy(sim_mat)
        adj_mat[adj_mat < threshold] = 0
        for i in range(adj_mat.shape[0]):
            adj_mat[i, i] = 0
            sim_mat[i, i] = 0
        # If a node doesn't have any connections, make edges to those with the highest similarity value.
        outliers = np.where(np.sum(adj_mat, axis=1) == 0)[0]
        if len(outliers) > 0:
            for outlier in outliers:
                largest_inds = sim_mat[outlier] == np.max(sim_mat[outlier])
                adj_mat[outlier, largest_inds] = np.max(sim_mat[outlier])
                adj_mat[largest_inds, outlier] = np.max(sim_mat[outlier])
        return adj_mat

    def select_first_batch(
        self,
        n_select,
        similarity_threshold=0.82,
        select_by="connection",
        tie_break="yield",
    ):
        """Selects the first batch of boronics to test by following a Butina-clustering-like procedure.
        0) Filter the adjacency score by the threshold
        1) Identifies nodes with highest sum of similarity scores after filtering.
            1-1) If there are multiple, select those with highest average yields across source halides
        2) Remove boronics that still have edges to the selected node
        3) Repeat this process until all nodes that have at least one edge are removed.

        Parameters
        ----------
        n_select : int
            Number of BBs to select.
        similarity_threshold : float
            Threshold value in reactivity similarity to use for whether an edge exists between two BBs.
        select_by : str {'connection', 'entropy'}
            The primary strategy for selecting the 'representative BBs'.
            • connection : utilize the network structure of the BBs
            • entropy : select the BBs with the highest entropy across the cores.
        tie_break : str {'yield', 'entropy'}
            How to break ties between BBs with same similarity strength.
            • yield: keep one with the highest average yield across source cores.
            • entropy : keep one where the results were the most different across the source cores.

        Returns
        -------
        selected_bb_inds : list of inds
            Indices corresponding to the list of SOURCE bb's not the whole list of bb's.
        """
        similarity_matrix = self.get_similarity_matrix(
            imputation="similarity", gephi_filename=None
        )
        # 0) Filter adjacency score by threshold
        adjacency_matrix = self._similarity_matrix_to_adjacency_matrix(
            similarity_matrix, similarity_threshold
        )
        self.adjacency_matrix = adjacency_matrix
        # row indices in the adjacency matrix.
        if select_by == "connection":
            node_candidates_for_selection = [
                i for i in range(adjacency_matrix.shape[0])
            ]  # if np.sum(adjacency_matrix[i, :])>1
            # print("AFTER REMOVING OUTLIERS", len(node_candidates_for_selection))
            selected_bb_inds = []
            neighbors_inds = []
            while len(node_candidates_for_selection) > 0:
                # 1)
                if tie_break == "yield":
                    summed_score = np.sum(
                        adjacency_matrix[node_candidates_for_selection, :], axis=1
                    )
                    max_sum_score = np.max(summed_score)
                    inds_with_highest_scores = np.where(summed_score == max_sum_score)[
                        0
                    ]
                    if len(inds_with_highest_scores) > 1:  # 1-1)
                        ind_selected = inds_with_highest_scores[
                            np.argmax(
                                np.nanmean(
                                    self.source_yield_array[
                                        :, inds_with_highest_scores
                                    ],
                                    axis=0,
                                )
                            )
                        ]
                    else:
                        ind_selected = inds_with_highest_scores[0]
                elif "entropy" in tie_break:
                    num_connections = np.sum(
                        adjacency_matrix[node_candidates_for_selection, :] > 0, axis=1
                    )
                    max_connections = np.max(num_connections)
                    inds_with_highest_scores = np.where(
                        num_connections == max_connections
                    )[0]
                    if len(inds_with_highest_scores) > 1:  # 1-1)
                        each_BB_reactivity = self.imputed_reactivity_array[
                            :, inds_with_highest_scores
                        ].T
                        entropy_vals = []
                        for i in range(len(inds_with_highest_scores)):
                            _, counts = np.unique(
                                each_BB_reactivity[i], return_counts=True
                            )
                            probs = counts / each_BB_reactivity.shape[1]
                            n_classes = np.count_nonzero(probs)
                            if n_classes <= 1:
                                entropy_vals.append(0)
                            else:
                                ent = 0
                                base = e
                                for p in probs:
                                    ent -= p * log(p, base)
                                entropy_vals.append(ent)
                        if tie_break == "max_entropy":
                            ind_selected = inds_with_highest_scores[
                                np.argmax(entropy_vals)
                            ]
                        elif tie_break == "min_entropy":
                            ind_selected = inds_with_highest_scores[
                                np.argmin(entropy_vals)
                            ]
                    else:
                        ind_selected = inds_with_highest_scores[0]
                ind_selected = node_candidates_for_selection[ind_selected]
                selected_bb_inds.append(ind_selected)
                neighbors_list = []
                # 2) Removing neighbors of selected boronic
                for ind_to_remove in np.where(adjacency_matrix[ind_selected, :] > 0)[0]:
                    if ind_to_remove in node_candidates_for_selection:
                        neighbors_list.append(ind_to_remove)
                        node_candidates_for_selection.remove(ind_to_remove)
                node_candidates_for_selection.remove(ind_selected)
                neighbors_inds.append(neighbors_list)
            selected_bb_inds = selected_bb_inds[:n_select]
            self.selected_bb_inds = selected_bb_inds
            self.neighbors_inds = neighbors_inds

        elif select_by == "entropy":
            entropy_by_BB = np.zeros(self.source_reactivity_array.shape[1])
            for i in range(self.source_reactivity_array.shape[1]):
                _, counts = np.unique(
                    self.source_reactivity_array[:, i], return_counts=True
                )
                probs = counts / self.source_reactivity_array.shape[0]
                n_classes = np.count_nonzero(probs)
                if n_classes <= 1:
                    entropy_by_BB[i] = 0
                else:
                    ent = 0
                    base = e
                    for p in probs:
                        ent -= p * log(p, base)
                    entropy_by_BB[i] = ent
            selected_bb_inds = np.argsort(entropy_by_BB)[-1 * n_select :]

        return selected_bb_inds

    def update_matrix_btw_BBs(
        self, first_selected_bb_inds, similarity_threshold, zero_same_factor=1
    ):
        """Updates the reactivity similarity matrix between building blocks
        after obtaining the results of the selected BBs with the new core.

        Parameters
        ---------
        first_selected_bb_inds : list of ints
            Indices of the BBs that were selected in the first batch.
        similarity_threshold : float

        zero_same_factor: float (0, 1]
            Weight impose on reactivity similarity score when two reactions are same by being negatives.

        Returns
        -------
        updated_similarity_matrix : np.ndarray of shape (n_BBs, n_BBs)
        """
        retrieved_target_reactivity = self.target_reactivity_array[
            first_selected_bb_inds
        ]
        self.similarities_of_source_cores_to_target_core = np.array(
            [
                self._compute_reactivity_similarity(
                    row, retrieved_target_reactivity, zero_same_factor=zero_same_factor
                )
                for row in self.imputed_reactivity_array[
                    :, first_selected_bb_inds
                ]  # source_reactivity_array
            ]
        )
        updated_similarity_matrix = np.zeros(
            (len(self.train_bb_inds), len(self.train_bb_inds))
        )
        for i in range(len(self.train_bb_inds)):
            updated_similarity_matrix[i, i + 1 :] = [
                self._compute_reactivity_similarity(
                    self.imputed_reactivity_array[:, i],
                    self.imputed_reactivity_array[:, x],
                    weights=self.similarities_of_source_cores_to_target_core,
                )
                for x in range(i + 1, len(self.train_bb_inds))
            ]
        updated_similarity_matrix += updated_similarity_matrix.T
        if similarity_threshold == 0:
            return updated_similarity_matrix
        else:
            return self._similarity_matrix_to_adjacency_matrix(
                updated_similarity_matrix, similarity_threshold
            )

    def select_second_batch(
        self,
        how_many_source_cores_to_consider,
        n_select,
        first_selected_bb_inds,
        zero_same_factor=0.5,
    ):
        """Based on the results of the first batch of experiments, greedily select the second batch.

        Parameters
        ----------
        how_many_source_halides_to_consider : int
            Number of 'similar' halides to consider in selecting the next set of boronics.
        zero_same_factor : float
            The weight of two values both being zero. The score will be multiplied by this value.
        n_select : int
            Number of building blocks to sample.
        first_selected_bb_inds : list of ints
            Indices of building blocks among the source set, selected in previous round.

        Returns
        -------
        selected_boronics_inds : list of ints
            Indices corresponding to self._all_boronics
        """
        remaining_inds = [
            x for x in range(len(self.train_bb_inds)) if x not in first_selected_bb_inds
        ]
        first_selected_classes = self.target_reactivity_array[first_selected_bb_inds]
        # same_bb_source_classes = self.source_reactivity_array[:, first_selected_bb_inds]
        same_bb_source_classes = self.imputed_reactivity_array[
            :, first_selected_bb_inds
        ]
        similarities = np.array(
            [
                self._compute_reactivity_similarity(
                    first_selected_classes,
                    source_row,
                    zero_same_factor=zero_same_factor,
                )
                for source_row in same_bb_source_classes
            ]
        )
        # print(similarities)
        cores_to_consider = np.argsort(similarities)[
            -1 * how_many_source_cores_to_consider :
        ]
        if how_many_source_cores_to_consider > 1:
            if self.mask is not None:
                weighted_classes = np.divide(
                    np.sum(
                        np.multiply(
                            self.imputed_reactivity_array[cores_to_consider, :],
                            similarities[cores_to_consider].reshape(-1, 1),
                        ),
                        axis=0,
                    ),
                    np.sum([similarities[x] for x in cores_to_consider]),
                )
                # Need to prioritize those with lower uncertainties
                # i.e., those with more yields not masked
                no_both_nan_yields = deepcopy(
                    self.source_yield_array[cores_to_consider]
                )
                no_both_nan_yields[
                    :, np.where(np.sum(np.isnan(no_both_nan_yields), axis=0) == 2)
                ] = 0
                weighted_classes += 0.01 * np.nanmean(no_both_nan_yields, axis=0)
                weighted_classes += 0.005 * np.nanstd(no_both_nan_yields, axis=0)
            else:  # use yield averages when full yields are available
                weighted_classes = np.sum(
                    np.multiply(
                        self.source_yield_array[cores_to_consider, :],
                        similarities[cores_to_consider].reshape(-1, 1),
                    ),
                    axis=0,
                ) - 0.5 * np.std(self.source_yield_array[cores_to_consider], axis=0)

        elif how_many_source_cores_to_consider == 1:
            weighted_classes = self.imputed_reactivity_array[
                remaining_inds, cores_to_consider
            ]
        sorted_inds = np.argsort(weighted_classes)[::-1]
        selected_bb_inds = []
        for ind in sorted_inds:
            if len(selected_bb_inds) < n_select:
                if ind in remaining_inds:
                    selected_bb_inds.append(ind)
        return selected_bb_inds

    def predict_remaining(
        self,
        how_many_source_cores_to_consider,
        first_selected_bb_inds,
        zero_same_factor=0.5,
    ):
        """Based on the results of the first batch of experiments, predict the outcomes of the remaining BBs with the new core.

        Parameters
        ----------
        how_many_source_halides_to_consider : int
            Number of 'similar' halides to consider in selecting the next set of boronics.
        first_selected_bb_inds : list of ints
            Indices of building blocks among the source set, selected in previous round.
        zero_same_factor : float
            The weight of two values both being zero. The score will be multiplied by this value.

        Returns
        -------
        weighted_classes : list of floats
            Averaged predicted class.
        """
        remaining_inds = [
            x for x in range(len(self.train_bb_inds)) if x not in first_selected_bb_inds
        ]
        first_selected_classes = self.target_reactivity_array[first_selected_bb_inds]
        # same_bb_source_classes = self.source_reactivity_array[:, first_selected_bb_inds]
        same_bb_source_classes = self.imputed_reactivity_array[
            :, first_selected_bb_inds
        ]
        similarities = np.array(
            [
                self._compute_reactivity_similarity(
                    first_selected_classes,
                    source_row,
                    zero_same_factor=zero_same_factor,
                )
                for source_row in same_bb_source_classes
            ]
        )
        print(similarities)
        cores_to_consider = np.argsort(similarities)[
            -1 * how_many_source_cores_to_consider :
        ]
        if how_many_source_cores_to_consider > 1:
            weighted_classes = np.sum(
                np.multiply(
                    # self.source_reactivity_array[cores_to_consider, :],
                    self.imputed_reactivity_array[cores_to_consider, :],
                    similarities[cores_to_consider].reshape(-1, 1),
                ),
                axis=0,
            ) / np.sum(similarities[cores_to_consider])
        elif how_many_source_cores_to_consider == 1:
            weighted_classes = self.imputed_reactivity_array[
                cores_to_consider, remaining_inds
            ]

        return weighted_classes
