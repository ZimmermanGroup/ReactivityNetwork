import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize
from sklearn.metrics import (
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from math import ceil
import seaborn as sns
# import scikit_posthocs as sp
import os
from rmap import *


class Analyzer:
    """Processes result files from second_selection.py for drawing plots and performing downstream analyses.

    Parameters
    ----------
    list_of_target_cores : list of int
        Target cores to draw together.
    list_of_result_filenames : list of str
        Filenames of the joblib files containing results.
    list_of_evaluation_names : list of str
        Names of evaluations that are being analyzed - used for labeling plots.
    """

    def __init__(
        self,
        list_of_target_cores,
        list_of_result_filenames,
        list_of_evaluation_names,
    ):
        if type(list_of_target_cores) == int:
            self.list_of_target_cores = [list_of_target_cores]
        elif type(list_of_target_cores) == list:
            self.list_of_target_cores = list_of_target_cores
        self.list_of_result_filenames = list_of_result_filenames
        self.list_of_evaluation_names = list_of_evaluation_names
        if len(self.list_of_result_filenames) != len(self.list_of_evaluation_names):
            raise ValueError(
                "list_of_target_cores and list_of_evaluation_names must have the same length."
            )

        self.score_dict = {
            "roc_auc": roc_auc_score,
            "auprc": average_precision_score,
        }

    def result_dicts(self, target_core_ind):
        "Loading the result dictionaries for a given core that resulted from second_selection.py"
        self._result_dicts = [
            joblib.load(os.path.join(f"saved_results/suzuki/{target_core_ind}", x))
            for x in self.list_of_result_filenames
        ]
        return self._result_dicts

    def _process_single_target_core(
        self, target_core_ind, metric, consider_middle_only=False, checkpoint=False
    ):
        if not checkpoint:
            sub_dict_to_plot = {"Models": [], "Score": []}
        else:
            sub_dict_to_plot = {
                "Models": [],
                "Reactivity sum": [],
                "Number of BB selections": [],
                "Score": [],
            }
        # Masking off the top and bottom 20%
        for eval_name, result_dict in zip(
            self.list_of_evaluation_names, self.result_dicts(target_core_ind)
        ):
            if len(result_dict["bootstrap_id"]) == len(
                np.unique(result_dict["bootstrap_id"])
            ):
                for target, proba, test_bb_inds, remaining_inds in zip(
                    result_dict["target"],
                    result_dict["proba"],
                    result_dict["test_bb_inds"],
                    result_dict["remaining_inds"],
                ):
                    # target here is binary
                    if consider_middle_only:
                        rmap = ReactivityMap(
                            None,
                            target_core_ind,
                            test_bb_inds,
                            dataset=SuzukiDataset(),
                            mask=None,
                        )
                        source_reactivity_sum = np.argsort(
                            np.sum(rmap.source_reactivity_array, axis=0)[remaining_inds]
                        )
                        inds_to_consider = [x for x in source_reactivity_sum[10:-10]]
                        sub_dict_to_plot["Score"].append(
                            self.score_dict[metric](
                                target[inds_to_consider],
                                proba.flatten()[inds_to_consider],
                            )
                        )
                    else:
                        sub_dict_to_plot["Score"].append(
                            self.score_dict[metric](target, proba.flatten())
                        )
                    if checkpoint:
                        proba_sorted = np.argsort(proba.flatten()[inds_to_consider])[
                            ::-1
                        ]
                        three_sets = [
                            [inds_to_consider[x] for x in proba_sorted[:5]],
                            [inds_to_consider[x] for x in proba_sorted[:10]],
                            [inds_to_consider[x] for x in proba_sorted[:15]],
                            [inds_to_consider[x] for x in proba_sorted[:20]],
                        ]
                        for num, top_set in enumerate(three_sets):
                            sub_dict_to_plot["Reactivity sum"].append(
                                np.sum(target[top_set])
                                / np.sum(
                                    np.sort(target[inds_to_consider])[-5 * (num + 1) :]
                                )  # precision
                            )
                            sub_dict_to_plot["Number of BB selections"].append(
                                5 * (num + 1)
                            )
                        sub_dict_to_plot["Models"].extend([eval_name] * len(three_sets))
                    else:
                        sub_dict_to_plot["Models"].append(eval_name)

            else:  # For random selections, first average the scores
                result_df = pd.DataFrame(result_dict)
                for bootstrap_id in result_df["bootstrap_id"].unique():
                    targets = result_df[result_df["bootstrap_id"] == bootstrap_id][
                        "target"
                    ].values
                    probas = result_df[result_df["bootstrap_id"] == bootstrap_id][
                        "proba"
                    ].values
                    scores = []
                    for target, proba in zip(targets, probas):
                        scores.append(self.score_dict[metric](target, proba.flatten()))

                    sub_dict_to_plot["Score"].append(np.mean(scores))
                    sub_dict_to_plot["Models"].append(eval_name)
        if checkpoint:
            del sub_dict_to_plot["Score"]
        return sub_dict_to_plot

    def prepare_score_dicts(self, metric, consider_middle_only=False, checkpoint=False):
        """Prepares a dictionary of scores for plotting."""
        if len(self.list_of_target_cores) == 1:
            dict_to_plot = self._process_single_target_core(
                self.list_of_target_cores[0], metric
            )
        else:
            dict_to_plot = []
            for target_core in self.list_of_target_cores:
                core_dict = self._process_single_target_core(
                    target_core, metric, consider_middle_only, checkpoint
                )
                sub_dict_to_plot = pd.DataFrame(core_dict)
                sub_dict_to_plot["Target Core"] = [
                    target_core
                ] * sub_dict_to_plot.shape[0]
                dict_to_plot.append(sub_dict_to_plot)
            dict_to_plot = pd.concat(dict_to_plot, ignore_index=True)
        return dict_to_plot
    
    def draw_barplot(
        self,
        metric,
        filename=None,
        ymin=None,
        ymax=None,
        colordict=None,
        consider_middle_only=False,
    ):
        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        dict_to_plot = self.prepare_score_dicts(
            metric, consider_middle_only=consider_middle_only, checkpoint=False
        )
        ax.set_ylabel(metric.upper(), fontdict={"fontsize": 10, "fontfamily": "Arial"})
        ax.set_yticklabels
        sns.barplot(
            data=dict_to_plot,
            x="Target Core",
            y="Score",
            hue="Models",
            ax=ax,
            palette=colordict,
            hue_order=self.list_of_evaluation_names,
        )
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.5)
        ax.legend(prop=fm.FontProperties(family="Arial", size=8), loc="lower right")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([round(x, 1) for x in np.arange(0.0, 1.01, 0.2)])
        ax.set_yticklabels(
            [round(x, 1) for x in np.arange(0.0, 1.01, 0.2)],
            fontdict={"fontsize": 8, "fontfamily": "Arial"},
        )
        ax.set_xticks(np.arange(len(self.list_of_target_cores)))
        ax.set_xticklabels(
            [x + 1 for x in self.list_of_target_cores],
            fontsize=8,
            fontfamily="Arial",
            fontweight="bold",
        )
        ax.set_xlabel(
            "Target Core", fontdict={"fontsize": 10, "fontfamily": "Arial"}
        )
        if len(self.list_of_target_cores) > 1:
            for x in range(len(self.list_of_target_cores)):
                if x < len(self.list_of_target_cores) - 1:
                    ax.axvline(x + 0.5, 0, 1, color="grey", linestyle="--", lw=0.5)
        if metric == "roc_auc":
            ax.axhline(0.5, 0, 1, color="grey", linestyle="--", lw=0.5, alpha=0.5)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300, bbox_inches="tight", format="svg")
            plt.close(fig)

    def draw_swarmplot(
        self,
        metric,
        filename=None,
        ymin=None,
        ymax=None,
        colordict=None,
        consider_middle_only=False,
    ):
        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        dict_to_plot = self.prepare_score_dicts(
            metric, consider_middle_only=consider_middle_only, checkpoint=False
        )
        ax.set_ylabel(metric.upper(), fontdict={"fontsize": 10, "fontfamily": "Arial"})
        sns.boxplot(
            data=dict_to_plot,
            x="Target Core",
            y="Score",
            hue="Models",
            whis=0.5,
            width=0.8,
            showcaps=True,
            boxprops={"facecolor": "None"},
            ax=ax,
            palette=colordict,
            dodge=True,
            linewidth=0.5,
            fliersize=0,
            linecolor="black",
            gap=0.25,
            hue_order=self.list_of_evaluation_names,
            legend=False,
        )
        if ymin is None and ymax is None:
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([round(x, 1) for x in np.arange(0.0, 1.01, 0.2)])
            ax.set_yticklabels(
                [round(x, 1) for x in np.arange(0.0, 1.01, 0.2)],
                fontdict={"fontsize": 8, "fontfamily": "Arial"},
            )
        if len(self.list_of_target_cores) == 1:  # Then the xticks are the models
            ax.set_xticks(np.arange(len(self.list_of_evaluation_names)))
            ax.set_xticklabels(
                self.list_of_evaluation_names, fontsize=8, fontfamily="Arial"
            )
            ax.set_xlabel("Models", fontdict={"fontsize": 10, "fontfamily": "Arial"})
            sns.swarmplot(data=dict_to_plot, x="Models", y="Score", size=3)
        else:  # Then the xticks are the target cores
            ax.set_xticks(np.arange(len(self.list_of_target_cores)))
            ax.set_xticklabels(
                [x + 1 for x in self.list_of_target_cores],
                fontsize=8,
                fontfamily="Arial",
                fontweight="bold",
            )
            ax.set_xlabel(
                "Target Core", fontdict={"fontsize": 10, "fontfamily": "Arial"}
            )

            if colordict is None:
                sns.swarmplot(
                    data=dict_to_plot,
                    x="Target Core",
                    y="Score",
                    hue="Models",
                    size=3,
                    ax=ax,
                    palette="viridis",
                    dodge=True,
                    hue_order=self.list_of_evaluation_names,
                )
            else:
                sns.swarmplot(
                    data=dict_to_plot,
                    x="Target Core",
                    y="Score",
                    hue="Models",
                    size=3,
                    ax=ax,
                    palette=colordict,
                    dodge=True,
                    hue_order=self.list_of_evaluation_names,
                    alpha=0.7,
                )
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.5)
        ax.legend(prop=fm.FontProperties(family="Arial", size=8), loc="lower right")

        if len(self.list_of_target_cores) > 1:
            for x in range(len(self.list_of_target_cores)):
                if x < len(self.list_of_target_cores) - 1:
                    ax.axvline(x + 0.5, 0, 1, color="grey", linestyle="--", lw=0.5)
        if metric == "roc_auc":
            ax.axhline(0.5, 0, 1, color="grey", linestyle="--", lw=0.5, alpha=0.5)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300, bbox_inches="tight", format="svg")
            plt.close(fig)

    def draw_checkpoints(self, filename=None, colordict=None):
        df_to_plot = self.prepare_score_dicts(
            "roc_auc", consider_middle_only=True, checkpoint=True
        )
        for target_core in df_to_plot["Target Core"].unique():
            fig, ax = plt.subplots(figsize=(2.7, 2))
            sns.lineplot(
                df_to_plot[df_to_plot["Target Core"] == target_core],
                x="Number of BB selections",
                y="Reactivity sum",
                hue="Models",
                marker="o",
                ax=ax,
                hue_order=self.list_of_evaluation_names,
                palette=colordict,
            )
            ax.set_xlim(3, 22)
            ax.set_xticks(np.arange(5, 22, 5))
            ax.set_xticklabels(
                [round(x, 0) for x in np.arange(5, 22, 5)],
                fontsize=8,
                fontfamily="arial",
            )
            ax.set_xlabel("Number of selections", fontsize=10, fontfamily="arial")
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.05, 0.2))
            ax.set_yticklabels(
                [round(x, 1) for x in np.arange(0, 1.05, 0.2)],
                fontsize=8,
                fontfamily="arial",
            )
            ax.set_ylabel("Portion of class-2", fontsize=10, fontfamily="arial")
            if target_core == 3:
                ax.legend(
                    bbox_to_anchor=(1.02, 0.95),
                    prop=fm.FontProperties(family="Arial", size=8),
                )
            else:
                ax.get_legend().remove()
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.5)
            plt.savefig(
                filename + f"_core{target_core+1}.svg",
                dpi=300,
                bbox_inches="tight",
                format="svg",
            )


def main(
    target_cores_to_draw_together,
    filenames_to_compare,
    evaluation_names,
    plotname=None,
    colordict=None,
    type="swarm",
    consider_middle_only=False,
    figsize_x=6.5
):
    """
    Parameters
    ----------
    target_cores_to_draw_together : list of ints
        Target cores to draw together.
    filenames_to_compare : list of str
        Filenames to compare.
    evaluation_names : list of str
        Names of evaluations that are being analyzed.
    """
    analyzer = Analyzer(
        target_cores_to_draw_together, filenames_to_compare, evaluation_names
    )
    if not os.path.exists("figures/eval_second"):
        os.mkdir("figures/eval_second")
    if type == "swarm":
        for metric in ["roc_auc", "auprc"]:
            plot_name = f"figures/eval_second/{plotname}_{metric}.svg"
            analyzer.draw_swarmplot(
                metric,
                filename=plot_name,
                colordict=colordict,
                consider_middle_only=consider_middle_only,
                figsize_x=figsize_x
            )
    elif type == "line":
        plot_name = f"figures/eval_second/{plotname}"
        analyzer.draw_checkpoints(filename=plot_name, colordict=colordict)
    elif type == "bar":
        for metric in ["roc_auc", "auprc"]:
            plot_name = f"figures/eval_second/{plotname}_{metric}.svg"
            analyzer.draw_barplot(
                metric,
                filename=plot_name,
                colordict=colordict,
                consider_middle_only=consider_middle_only,
            )


if __name__ == "__main__":
    target_cores_to_draw_together = [0, 1, 3, 7, 8]

    ######################## Figures 6, S26, S27 ########################
    filenames_to_compare = [
        "rmap_6_baseline.joblib",
        "rmap_6_rfc_combined.joblib",
        "rmap_6_rmap.joblib"
    ]
    evaluation_names = [
        "Baseline",
        "RFC",
        "LP (Rmap)",
    ] # For the main text
    colors = sns.color_palette("colorblind", 8)
    colordict = {
        "LP (Rmap)":colors[1],
        "RFC":colors[0],
        "Baseline":colors[-1]
    }
    main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "Figure6C", colordict=colordict, type="line", consider_middle_only=True) # Also leads to S30
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "Figure6B", colordict=colordict, consider_middle_only=True) # Also leads to S26
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS27", colordict=colordict, consider_middle_only=False)
    ########################################################################

    ######################## Figure S28 ########################
    # filenames_to_compare = [
    #     "rmap_6_baseline_ntest0_nbootstrap1.joblib",
    #     "rmap_6_rfc_combined_ntest0_nbootstrap1.joblib",
    #     "rmap_6_rmap_ntest0_nbootstrap1.joblib"
    # ]
    # evaluation_names = [
    #     "Baseline",
    #     "RFC",
    #     "LP (Rmap)",
    # ] # For the main text
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "LP (Rmap)":colors[1],
    #     "RFC":colors[0],
    #     "Baseline":colors[-1]
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS28", colordict=colordict, consider_middle_only=False, type="bar")
    ########################################################################

    # ######################## Figure S21 ########################
    # filenames_to_compare = [
    #     "uncertainty_6_rfc_target.joblib",
    #     "desc_cluster_6_rfc_target.joblib",
    #     "uncertainty_6_rfc_combined.joblib",
    #     "desc_cluster_6_rfc_combined.joblib",
    # ]
    # evaluation_names = [
    #     "Uncertainty+Target",
    #     "Desc cluster+Target",
    #     "Uncertainty+Combined",
    #     "Desc cluster+Combined",
    # ]
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS20", consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S22 ########################
    # filenames_to_compare =[
    #     "rmap_6_rfc_combined.joblib",
    #     "desc_cluster_6_rfc_combined.joblib",
    # ]
    # evaluation_names = [
    #     "Rmap",
    #     "k-means"
    # ]
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "Rmap":colors[1],
    #     "k-means":colors[0]
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS21", colordict=colordict, consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S23 ########################
    # filenames_to_compare = [
    #     f"random_6_rmap.joblib",
    #     f"modularity_6_rmap.joblib",
    #     f"rmap_6_rmap.joblib",
    # ]  # For comparing selection methods on label propagation performance
    # evaluation_names = [
    #     "Random",
    #     "Modularity",
    #     "Rmap",
    # ]
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS22", consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S24 ########################
    # filenames_to_compare = [
    #     f"rmap_6_rmap.joblib",
    #     f"rmap_6_gcn_ohe.joblib",
    #     f"rmap_6_gcn.joblib",
    # ]  # For comparing label propagation and GCN performance
    # evaluation_names = [
    #     "LP",
    #     "GCN (OHE)",
    #     "GCN (desc)",
    # ]
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS23", consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S25 ########################
    # filenames_to_compare = [
    #     "rmap_6_rmap.joblib",
    #     "rmap_6_rmap_tanimoto.joblib",
    #     "rmap_6_knn.joblib",
    #     "rmap_6_svm.joblib",
    # ]  # For comparing label propagation and GCN performance
    # evaluation_names = [
    #     "LP (Rmap)",
    #     "LP (Tanimoto)",
    #     "kNN",
    #     "SVM"
    # ]
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "LP (Rmap)":colors[1],
    #     "LP (Tanimoto)":colors[2],
    #     "kNN": colors[3],
    #     "SVM": colors[4],
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureSXX", consider_middle_only=False, colordict=colordict)
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureSXX_middle", consider_middle_only=True, colordict=colordict)
    # ########################################################################