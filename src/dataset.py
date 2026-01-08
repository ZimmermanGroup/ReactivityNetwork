import numpy as np
import pandas as pd
import os
from abc import ABC
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from itertools import product

FILEDIR = "data"
RAW_DATA_FILENAMES = [
    os.path.join(FILEDIR, f"{x}_final_report.csv")
    for x in ["008b", "010", "011", "012", "013"]
]  # The filenames should be changed
CLASS_THRESHOLDS = [20, 40]


class Dataset(ABC):
    """Base class for reaction datasets.
    Following the convention in sklearn where the dataset splitting etc is determined separately, only prepares full data.

    Parameters
    ----------
    class_thresholds: list of ints
        Yield threshold values to use to define the classes for classification.
    for_conventional_models : bool
        Whether the datasets will be used for building conventional regressors or classifiers.
    """

    def __init__(self, class_thresholds, for_conventional_models):
        self.class_thresholds = class_thresholds
        self.for_conventional_models = for_conventional_models

        self.core_smiles = []
        self.bb_smiles = []
        self.core_descriptors = np.zeros((4, 6))
        self.bb_descriptors = np.zeros((4, 6))
        # Specific datasets should define the following attributes
        # self.df, self.core_smiles, self.bb_smiles, self.core_descriptors, self.bb_descriptors

    @property
    def yield_array(self):
        """Prepares the full yield array.
        if not self.for_conventional_models: rows=core, columns=bb
        else: index=core_index*n_bb + bb_index"""
        pass

    @property
    def reactivity_array(self):
        """Prepares array of outcome classes of the target halide for downstream reactivity clustering analysis.

        Parameters
        ----------
        None

        Returns
        -------
        target_reactivity_array : np.ndarray of shape (n_boronics, 1)
            Yield classes of reactions of halide of interest (target).
        """
        self._reactivity_array = np.digitize(
            self.yield_array, self.class_thresholds, right=True
        )
        return self._reactivity_array

    def _smiles_to_numpy_fp_array(self, nbits, rad, list_of_smiles):
        """Generates fingerprints as numpy arrays from smiles for input for ML predictions.

        Parameters
        ----------
        fpgen: rdkit FP generator object.
            Fingerprint generator with specified parameters.
        list_of_smiles : list of strings
            smiles strings to convert into fingerprints.

        Returns
        -------
        arr : np.ndarray of shape (n_bits, )
            Resulting fingerprint.
        """
        fp_array = np.zeros((len(list_of_smiles), nbits))
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=rad, fpSize=nbits)
        for i, smiles in enumerate(list_of_smiles):
            fp = fpgen.GetCountFingerprint(Chem.MolFromSmiles(smiles))
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_array[i] = arr
        return fp_array

    @property
    def X_fp(self):
        """Prepares Morgan Count FP array of boronics reacting with either source or target halide(s).

        Parameters
        ----------
        nbits : int
            Number of fingerprint bits
        rad : int
            Radius of the sub-fragment for the fingerprint to use.

        Returns
        -------
        fp_array : np.ndarray of shape (n_core_molecules, n_building_blocks, 2*nbits) if not self.for_conventional_models
        """
        nbits = 1024
        rad = 2
        bb_fp_array = self._smiles_to_numpy_fp_array(nbits, rad, self.bb_smiles)
        core_array = self._smiles_to_numpy_fp_array(nbits, rad, self.core_smiles)

        self._X_fp = np.zeros((core_array.shape[0], bb_fp_array.shape[0], 2 * nbits))
        for i, j in product(range(core_array.shape[0]), range(bb_fp_array.shape[0])):
            self._X_fp[i, j, :] = np.concatenate(
                (core_array[i].flatten(), bb_fp_array[j].flatten())
            )

        return self._X_fp

    @property
    def X_desc(self):
        """Prepares arrays of physical descriptors.

        Parameters
        ----------
        None

        Returns
        -------
        desc_array : np.ndarray as the same shape as X_fp, except for the number of descriptors
        """
        self._X_desc = np.zeros(
            (
                self.core_descriptors.shape[0],
                self.bb_descriptors.shape[0],
                self.core_descriptors.shape[1] + self.bb_descriptors.shape[1],
            )
        )
        for i, j in product(
            range(self.core_descriptors.shape[0]), range(self.bb_descriptors.shape[0])
        ):
            self._X_desc[i, j, :] = np.concatenate(
                (self.core_descriptors[i].flatten(), self.bb_descriptors[j].flatten())
            )
        return self._X_desc

    @property
    def X_random(self):
        """Prepares array of random descriptors for adversarial control purposes.

        Parameters
        ----------
        None

        Returns
        -------
        X_random : np.ndarray as same shape as X_desc.
        """
        np.random.seed(42)
        self._X_random = np.random.rand(*self.X_desc.shape)
        return self._X_random

    @property
    def X_ohe(self):
        """Prepares array of one hot encodings for adversarial control purposes.

        Parameters
        ----------
        None

        Returns
        -------
        X_ohe : np.ndarray with shape (n_cores, n_bbs, n_cores+n_bb_s).
        """
        self._X_ohe = np.zeros(
            (
                len(self.core_smiles),
                len(self.bb_smiles),
                len(self.core_smiles) + len(self.bb_smiles),
            )
        )
        for i in range(len(self.core_smiles)):
            for j in range(len(self.bb_smiles)):
                self._X_ohe[i, j, i] = 1
                self._X_ohe[i, j, i + j] = 1
        return self._X_ohe


class SuzukiDataset(Dataset):
    """Class for the Suzuki dataset prepared by Relay Tx.
    For this dataset, we consider bromides as 'core' and boronic acid and esters as 'building blocks'.

    Parameters
    ----------
    class_thresholds : list of ints
        Yield threshold values to use to define the classes for classification.
    for_conventional_models : bool
        Whether the datasets will be used for building conventional regressors or classifiers.
    keepPhBr : bool
        Whether to include PhBr in the dataset. See dataset_analysis.ipynb and SI for details.
    """

    def __init__(
        self,
        class_thresholds=CLASS_THRESHOLDS,
        for_conventional_models=False,
        keepPhBr=False,
        from_notebook=False,
    ):
        """Aggregates ALL the raw data dataframes into a single dataframe.
        1) Multiple yield values are aggregated such that
        1-1) if one out of two total entries is positive --> keep the positive value.
        1-2) if there's more than two entries --> keep the median value across non-zero values.
        * Stoichiometry was adjusted after 8b - for the few 'control' substrate pairs, got median yields from plates 10-13.
        """
        super().__init__(class_thresholds, for_conventional_models)
        self.keepPhBr = keepPhBr

        path8b = "data/suzuki/008b_final_report.csv"
        if from_notebook:
            path8b = "../" + path8b
        # First need to filter off other reaction conditions from 8b
        df_8b = pd.read_csv(
            path8b,
            usecols=[
                "reactant_1&name",
                "reactant_1&smiles",
                "reactant_2&name",
                "reactant_2&smiles",
                "catalyst_1&name",
                "base_1&name",
                "solvent_2&name",
                "suzuki_product_CAD_yield",
            ],
        )
        df_8b_interest = df_8b[
            (df_8b["catalyst_1&name"] == "XPhos Pd G4")
            & (df_8b["solvent_2&name"] == "dioxane")
        ]

        # Combining the datasets together
        if not from_notebook:
            dfs = [df_8b_interest.iloc[:, [1, 2, 3, 4, -1]]] + [
                pd.read_csv(
                    f"data/suzuki/{x}_final_report.csv",
                    usecols=[
                        "reactant_1&name",
                        "reactant_1&smiles",
                        "reactant_2&name",
                        "reactant_2&smiles",
                        "suzuki_product_CAD_yield",
                    ],
                )
                for x in ["010", "011", "012", "013"]
            ]
        else:
            dfs = [df_8b_interest.iloc[:, [1, 2, 3, 4, -1]]] + [
                pd.read_csv(
                    f"../data/suzuki/{x}_final_report.csv",
                    usecols=[
                        "reactant_1&name",
                        "reactant_1&smiles",
                        "reactant_2&name",
                        "reactant_2&smiles",
                        "suzuki_product_CAD_yield",
                    ],
                )
                for x in ["010", "011", "012", "013"]
            ]
        raw_df = pd.concat(dfs, ignore_index=True)
        raw_df["plate"] = (
            ["008b"] * df_8b_interest.shape[0]
            + ["010"] * 384
            + ["011"] * 384
            + ["012"] * 384
            + ["013"] * 384
        )
        self._raw_df = raw_df

        # Keeping halides that were surveyed with ALL boronics
        all_halides = raw_df["reactant_1&smiles"].unique()
        self.core_smiles = []
        for halide in all_halides:
            if (
                len(
                    raw_df[raw_df["reactant_1&smiles"] == halide][
                        "reactant_2&smiles"
                    ].unique()
                )
                >= 69
            ):
                self.core_smiles.append(halide)
        if not self.keepPhBr:
            self.core_smiles = [x for x in self.core_smiles if x != "BrC1=CC=CC=C1"]
        self.bb_smiles = raw_df[raw_df["reactant_1&smiles"] == self.core_smiles[0]][
            "reactant_2&smiles"
        ].unique()

        # Looking at substrate pairs where reactions were done across Plates 8b and 10-12.
        plate8b_unique_pairs = raw_df[raw_df["plate"] == "008b"][
            ["reactant_1&smiles", "reactant_2&smiles"]
        ].drop_duplicates()
        substrates_in_8_to_12 = []
        for i, row in plate8b_unique_pairs.iterrows():
            if row.iloc[0] in self.core_smiles and row.iloc[1] in self.bb_smiles:
                if (
                    len(
                        raw_df[
                            (raw_df["reactant_1&smiles"] == row.iloc[0])
                            & (raw_df["reactant_2&smiles"] == row.iloc[1])
                        ]["plate"].values
                    )
                    > 1
                ):
                    substrates_in_8_to_12.append((row.iloc[0], row.iloc[1]))

        # Aggregating yields
        each_row = []
        count1 = 0
        count2 = 0
        for halide_smiles, boronic_smiles in product(self.core_smiles, self.bb_smiles):
            sub_df = raw_df[
                (raw_df["reactant_1&smiles"] == halide_smiles)
                & (raw_df["reactant_2&smiles"] == boronic_smiles)
            ]
            if sub_df.shape[0] == 1:
                each_row.append(sub_df.values.tolist()[0])
                if count1 == 0:
                    count1 += 1
            else:
                if (halide_smiles, boronic_smiles) not in substrates_in_8_to_12:
                    y_vals = sub_df["suzuki_product_CAD_yield"].to_numpy()
                else:
                    y_vals = sub_df[sub_df["plate"].isin(["010", "011", "012", "013"])][
                        "suzuki_product_CAD_yield"
                    ].to_numpy()
                pos_yvals = y_vals[np.where(y_vals > 0)]
                if len(pos_yvals) == 0:
                    y = 0
                else:
                    y = np.median(pos_yvals)
                row = list(sub_df.iloc[0, :-1].values) + [y]
                if count2 == 0:
                    count2 += 1
                each_row.append(row)
        self.df = pd.DataFrame(each_row, columns=sub_df.columns).iloc[:, :-1]

        # Preparing boronic descriptor arrays
        if not from_notebook:
            boronics_descriptors = pd.read_csv("data/suzuki/boronic_descriptors.csv")
        else:
            boronics_descriptors = pd.read_csv("../data/suzuki/boronic_descriptors.csv")
        boronic_desc = np.zeros(
            (boronics_descriptors.shape[0], boronics_descriptors.shape[1] - 3)
        )
        for i, row in boronics_descriptors.iterrows():
            boronic_desc[list(self.bb_smiles).index(row["reactant_2&smiles"]), :] = row[
                1:-2
            ]
        self.bb_descriptors = boronic_desc

        if not from_notebook:
            halide_descriptors = pd.read_csv("data/suzuki/halide_descriptors.csv")
        else:
            halide_descriptors = pd.read_csv("../data/suzuki/halide_descriptors.csv")
        halide_desc = np.zeros(
            (halide_descriptors.shape[0], halide_descriptors.shape[1] - 1)
        )
        halide_desc = np.vstack(
            tuple(
                [
                    halide_descriptors[halide_descriptors["reactant_1&smiles"] == x]
                    .iloc[:, 1:]
                    .to_numpy()
                    for x in self.core_smiles
                ]
            )
        )
        self.core_descriptors = halide_desc

    def __str__(self):
        return "suzuki"

    @property
    def yield_array(self):
        """Prepares the full yield array.
        if not self.for_conventional_models: rows=core, columns=bb
        else: index=core_index*n_bb + bb_index"""
        if self.for_conventional_models:
            return self.df["suzuki_product_CAD_yield"].to_numpy()
        else:
            return (
                self.df["suzuki_product_CAD_yield"]
                .to_numpy()
                .reshape((len(self.core_smiles), len(self.bb_smiles)))
            )
