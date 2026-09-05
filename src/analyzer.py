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
                "Number of hits": [],
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
                        # source_reactivity_sum = np.argsort(
                        #     np.sum(rmap.source_reactivity_array, axis=0)[remaining_inds]
                        # )
                        source_yield_sum = np.argsort(
                            np.sum(rmap.source_yield_array, axis=0)[remaining_inds]
                        )
                        inds_to_consider = [x for x in source_yield_sum[10:-10]]
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
                            sub_dict_to_plot["Number of hits"].append(
                                np.sum(target[top_set])
                            )
                            # sub_dict_to_plot["Reactivity sum"].append(
                            #     np.sum(target[top_set])
                            #     / np.sum(
                            #         np.sort(target[inds_to_consider])[-5 * (num + 1) :]
                            #     )  # precision
                            # )
                            sub_dict_to_plot["Number of BB selections"].append(
                                5 * (num + 1)
                            )
                        sub_dict_to_plot["Models"].extend([eval_name] * len(three_sets))
                    else:
                        sub_dict_to_plot["Models"].append(eval_name)

            else:  # For random selections, first average the scores
                # for k , v in result_dict.items():
                #     print(k, len(v))
                copy_result_dict = {k:v for k, v in result_dict.items() if k not in ["LP thresholds", "num components", "pilot_thresholds", "Number of pilots"]}

                result_df = pd.DataFrame(copy_result_dict)
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
                if consider_middle_only == "together":
                    core_dict1 = self._process_single_target_core(
                        target_core, metric, False, checkpoint
                    )
                    core_dict2 = self._process_single_target_core(
                        target_core, metric, True, checkpoint
                    )
                    core_dict = {
                        "Models":core_dict1["Models"] + core_dict2["Models"],
                        "Score":core_dict1["Score"] + core_dict2["Score"],
                        "Evaluation":["All BBs"]*len(core_dict1["Score"]) + ["Intermediate"]*len(core_dict2["Score"])
                    }
                else :
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
        # ax.set_yticklabels
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

    def draw_two_barplots_together(
        self,
        metric,
        filename=None,
        ymin=None,
        ymax=None,
        colordict=None,
    ):
        fig, ax = plt.subplots(figsize=(6.7, 2.5), tight_layout=True, sharey=True, ncols=2)
        dict_to_plot = self.prepare_score_dicts(
            metric, consider_middle_only="together", checkpoint=False
        )
        df_to_plot = pd.DataFrame(dict_to_plot)
        ax[0].set_ylabel(metric.upper(), fontdict={"fontsize": 10, "fontfamily": "Arial"})
        for i, (eval_name, sub_df) in enumerate(df_to_plot.groupby("Evaluation")):
            sns.barplot(
                data=sub_df,
                x="Target Core",
                y="Score",
                hue="Models",
                ax=ax[i],
                palette=colordict,
                hue_order=self.list_of_evaluation_names,
            )
            for axis in ["top", "bottom", "left", "right"]:
                ax[i].spines[axis].set_linewidth(1.5)
            if i == 1:
                ax[i].legend(prop=fm.FontProperties(family="Arial", size=8), loc="lower right")
            ax[i].set_ylim(0.0, 1.0)
            ax[i].set_yticks([round(x, 1) for x in np.arange(0.0, 1.01, 0.2)])
            ax[i].set_yticklabels(
                [round(x, 1) for x in np.arange(0.0, 1.01, 0.2)],
                fontdict={"fontsize": 8, "fontfamily": "Arial"},
            )
            ax[i].set_xticks(np.arange(len(self.list_of_target_cores)))
            ax[i].set_xticklabels(
                [x + 1 for x in self.list_of_target_cores],
                fontsize=8,
                fontfamily="Arial",
                fontweight="bold",
            )
            ax[i].set_xlabel(
                "Target Core", fontdict={"fontsize": 10, "fontfamily": "Arial"}
            )
            if len(self.list_of_target_cores) > 1:
                for x in range(len(self.list_of_target_cores)):
                    if x < len(self.list_of_target_cores) - 1:
                        ax[i].axvline(x + 0.5, 0, 1, color="grey", linestyle="--", lw=0.5)
            if metric == "roc_auc":
                ax[i].axhline(0.5, 0, 1, color="grey", linestyle="--", lw=0.5, alpha=0.5)
        ax[0].get_legend().remove()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300, bbox_inches="tight", format="svg")
            plt.close(fig)

    def draw_two_swarmplots_together(
        self,
        metric,
        filename=None,
        ymin=None,
        ymax=None,
        colordict=None
    ):
        fig, ax = plt.subplots(figsize=(6.7, 2.5), tight_layout=True, sharey=True, ncols=2)
        dict_to_plot = self.prepare_score_dicts(
            metric, consider_middle_only="together", checkpoint=False
        )
        df_to_plot = pd.DataFrame(dict_to_plot)
        ax[0].set_ylabel(metric.upper(), fontdict={"fontsize": 10, "fontfamily": "Arial"})
        evaluation_linecolors = {
            "All BBs":"black",
            "Intermediate":"black"
        }
        evaluation_markers = {
            "All BBs":"o",
            "Intermediate":"^"
        }
        if colordict is None :
            colordict = "viridis"
        for i, (eval_name, sub_df) in enumerate(df_to_plot.groupby("Evaluation")):
            ax[i].set_ylim(0.0, 1.0)
            ax[i].set_yticks([round(x, 1) for x in np.arange(0.0, 1.01, 0.2)])
            if i == 0:
                ax[i].set_yticklabels(
                    [round(x, 1) for x in np.arange(0.0, 1.01, 0.2)],
                    fontdict={"fontsize": 8, "fontfamily": "Arial"},
                )
            line_color = evaluation_linecolors[eval_name]
            sns.boxplot(
                data=sub_df,
                x="Target Core",
                y="Score",
                hue="Models",
                whis=0.5,
                width=0.8,
                showcaps=True,
                boxprops={"facecolor": "None"},
                ax=ax[i],
                palette=colordict,
                dodge=True,
                linewidth=0.5,
                fliersize=0,
                linecolor=line_color,
                gap=0.15,
                hue_order=self.list_of_evaluation_names,
                legend=False,
            )
            sns.swarmplot(
                data=sub_df,
                x="Target Core",
                y="Score",
                hue="Models",
                size=3,
                ax=ax[i],
                palette=colordict,
                dodge=True,
                hue_order=self.list_of_evaluation_names,
                marker=evaluation_markers[eval_name]
            )
        
            ax[i].set_xticks(np.arange(len(self.list_of_target_cores)))
            ax[i].set_xticklabels(
                [x + 1 for x in self.list_of_target_cores],
                fontsize=8,
                fontfamily="Arial",
                fontweight="bold",
            )
            ax[i].set_xlabel(
                "Target Core", fontdict={"fontsize": 10, "fontfamily": "Arial"}
            )

            for axis in ["top", "bottom", "left", "right"]:
                ax[i].spines[axis].set_linewidth(1.5)
        ax[1].legend(prop=fm.FontProperties(family="Arial", size=8), loc="lower right")
        ax[0].get_legend().remove()

        if len(self.list_of_target_cores) > 1:
            for x in range(len(self.list_of_target_cores)):
                if x < len(self.list_of_target_cores) - 1:
                    for i in [0, 1]:
                        ax[i].axvline(x + 0.5, 0, 1, color="grey", linestyle="--", lw=0.5)
        if metric == "roc_auc":
            for i in [0,1]:
                ax[i].axhline(0.5, 0, 1, color="grey", linestyle="--", lw=0.5, alpha=0.5)
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
        fig, ax = plt.subplots(figsize=(6.5, 2.5), tight_layout=True)
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
                y="Number of hits",
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
            if consider_middle_only == "together":
                analyzer.draw_two_swarmplots_together(
                    metric,
                    filename=plot_name,
                    colordict=colordict
                )
            else :
                analyzer.draw_swarmplot(
                    metric,
                    filename=plot_name,
                    colordict=colordict,
                    consider_middle_only=consider_middle_only,
                    # figsize_x=figsize_x
                )
    elif type == "line":
        plot_name = f"figures/eval_second/{plotname}"
        analyzer.draw_checkpoints(filename=plot_name, colordict=colordict)
    elif type == "bar":
        for metric in ["roc_auc", "auprc"]:
            plot_name = f"figures/eval_second/{plotname}_{metric}.svg"
            if consider_middle_only == "together":
                analyzer.draw_two_barplots_together(
                    metric,
                    filename=plot_name,
                    colordict=colordict
                )
            else :
                analyzer.draw_barplot(
                    metric,
                    filename=plot_name,
                    colordict=colordict,
                    consider_middle_only=consider_middle_only,
                )


if __name__ == "__main__":
    target_cores_to_draw_together = [0, 1, 3, 7, 8]

    ######################## Figures 6B ########################
    # filenames_to_compare = [
    #     "rmap_cv_6_baseline.joblib",
    #     "rmap_cv_6_rfc_combined.joblib",
    #     "rmap_cv_6_rmap_cv.joblib"
    # ]
    # evaluation_names = [
    #     "Baseline",
    #     "RFC",
    #     "LP",
    # ] # For the main text
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "LP":colors[1],
    #     "RFC":colors[0],
    #     "Baseline":colors[-1]
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "Figure6B_r1", colordict=colordict, consider_middle_only="together") # Also leads to S30
    ########################################################################

        
    ######################### Figure S21 ########################
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
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS21_r1", colordict=None, consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S22 ########################
    # filenames_to_compare =[
    #     "rmap_cv_6_rfc_combined.joblib",
    #     "desc_cluster_6_rfc_combined.joblib",
    # ]
    # evaluation_names = [
    #     "RNet",
    #     "Desc cluster"
    # ]
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "RNet":colors[1],
    #     "Desc cluster":colors[0]
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS22_r1", colordict=colordict, consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S23 ########################
    # filenames_to_compare = [
    #     f"random_6_rmap_predefined.joblib",
    #     f"modularity_6_rmap_predefined.joblib",
    #     f"rmap_6_rmap_predefined.joblib",
    # ]  # For comparing selection methods on label propagation performance
    # evaluation_names = [
    #     "Random",
    #     "Modularity",
    #     "RNet",
    # ]
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS23", consider_middle_only=False)
    # ########################################################################

    # ######################## Figure S24 ########################
    # filenames_to_compare = [
    #     f"rmap_cv_6_rmap_predefined.joblib",
    #     f"rmap_cv_6_gcn_ohe.joblib",
    #     f"rmap_cv_6_gcn.joblib",
    # ]  # For comparing label propagation and GCN performance
    # evaluation_names = [
    #     "LP",
    #     "GCN (OHE)",
    #     "GCN (desc)",
    # ]
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS24", consider_middle_only="together")
    # ########################################################################

    ######################## Figure S25 ########################
    # filenames_to_compare = [
    #     "rmap_cv_6_rmap_cv.joblib",
    #     "rmap_6_rmap_cv.joblib",
    #     "rmap_6_rmap_predefined.joblib"
    # ]
    # evaluation_names = [
    #     "Both CV",
    #     "Prediction CV",
    #     "No CV",
    # ] # For the main text
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "Both CV":colors[1],
    #     "Prediction CV":colors[0],
    #     "No CV":colors[-1]
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureSXXX_CVcomparison", colordict=colordict, consider_middle_only="together") # Also leads to S30
    
    ######################## Figure S26 ########################
    # filenames_to_compare = [
    #     "rmap_cv_6_rmap_cv.joblib",
    #     "rmap_cv_6_rmap_tanimoto_cv.joblib",
    #     "rmap_cv_6_knn_cv.joblib",
    #     "rmap_cv_6_svm_cv.joblib",
    # ]  # For comparing label propagation and GCN performance
    # evaluation_names = [
    #     "LP (RNet)",
    #     "LP (Tanimoto)",
    #     "kNN",
    #     "SVM"
    # ]
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "LP (RNet)":colors[1],
    #     "LP (Tanimoto)":colors[2],
    #     "kNN": colors[3],
    #     "SVM": colors[4],
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureSXX", consider_middle_only=False, colordict=colordict)
    ########################################################################

    ######################## Figure S34 - entire dataset ########################
    # filenames_to_compare = [
    #     "rmap_cv_6_baseline_ntest0_nbootstrap1.joblib",
    #     "rmap_cv_6_rfc_combined_ntest0_nbootstrap1.joblib",
    #     "rmap_cv_6_rmap_cv_ntest0_nbootstrap1.joblib"
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
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS28_EntireDataset", colordict=colordict, type="bar", consider_middle_only="together")

    ######################## Figure S40 - aggregation scheme analysis ########################
    # for agg in ["median", "average", "maximum"]:
    #     filenames_to_compare = [
    #         f"rmap_cv_6_baseline_includeZero_{agg}.joblib",
    #         f"rmap_cv_6_rmap_cv_includeZero_{agg}.joblib",
    #     ]
    #     if agg == "maximum":
    #         lp_label = f"LP ({agg})"
    #     else :
    #         lp_label = f"LP (all {agg})"
    #     evaluation_names = [
    #         "Baseline",
    #         lp_label,
    #     ]
    #     colors = sns.color_palette("colorblind", 8)
    #     colordict = {
    #         "Baseline":colors[1],
    #         lp_label:colors[2],
    #     }
    #     main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, f"FigureS_all{agg}", consider_middle_only="together", colordict=colordict)

    ######################## Figure SX ########################
    # filenames_to_compare = [
    #     "rmap_6_rmap_predefined.joblib",
    #     "rmap_6_rmap_predefined_eps0.joblib",
    # ]  # Epsilon sensitivity analysis
    # evaluation_names = [
    #     "eps=0.001",
    #     "eps=0",
    # ]
    # colors = sns.color_palette("colorblind", 8)
    # colordict = {
    #     "eps=0.001":colors[1],
    #     "eps=0":colors[2],
    # }
    # main(target_cores_to_draw_together, filenames_to_compare, evaluation_names, "FigureS_eps", consider_middle_only=False, colordict=colordict)