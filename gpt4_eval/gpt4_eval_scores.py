import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pycountry
import time

# from utils import *
from utils import TASKS, AR, EN, GPT4_PROMPT_TYPE
from utils import get_comparison_out_file_path, gpt4_comp_path


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name


def survey(args, results):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    fig_name_model1 = args.model1 if not args.fig_name_model1 else args.fig_name_model1
    fig_name_model2 = args.model2 if not args.fig_name_model2 else args.fig_name_model2

    category_names = [fig_name_model1, "Tie", fig_name_model2]

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    text_colors = ['white', 'darkgrey', 'white']
    colors = ["#00D2AA", '#FFFFCB', "#00afff"]

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        # rects = ax.barh(labels, widths, left=starts, height=0.5,
        #                 label=colname, color=color)
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=colors[i % 3])

        r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        text_color = text_colors[i % 3]
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    if args.cross:
        plt.savefig(
            os.path.join(gpt4_comp_path,
                         f'{args.model1}_vs_{args.model2}_{args.task}_{args.gpt4_prompt_type}_cross.png'),
            bbox_inches='tight', dpi=300, transparent=True)
    else:
        plt.savefig(
            os.path.join(gpt4_comp_path, f'{args.model1}_vs_{args.model2}_{args.task}_{args.gpt4_prompt_type}.png'),
            bbox_inches='tight', dpi=300, transparent=True)
    return fig, ax


def get_winning_rate_avg(df1, df2, model1, model2):
    # avg scores
    df1[f"avg_{model1}_score"] = (df1[f"{model1}_score"] + df2[f"{model1}_score"]) / 2
    df2[f"avg_{model2}_score"] = (df1[f"{model2}_score"] + df2[f"{model2}_score"]) / 2
    # avg results
    lwin = sum(df1[f"avg_{model1}_score"] > df2[f"avg_{model2}_score"])
    tie = sum(df1[f"avg_{model1}_score"] == df2[f"avg_{model2}_score"])
    rwin = sum(df1[f"avg_{model1}_score"] < df2[f"avg_{model2}_score"])

    return lwin, tie, rwin


def get_winning_rate_consistent(df1, df2, model1, model2):
    # trend
    df1["trend"] = df1[f"{model1}_score"] - df1[f"{model2}_score"]
    df2["trend"] = df2[f"{model1}_score"] - df2[f"{model2}_score"]

    condition = (np.sign(df1['trend']) == np.sign(df2['trend']))  # if both direction scores are same
    condition1 = ((df1['trend'] != 0) & (df2['trend'] == 0))  # if df2 is tied but df1 is not condition
    condition2 = ((df2['trend'] != 0) & (df1['trend'] == 0))  # if df1 is tied but df2 is not condition

    inconsistent_condition = (
                (np.sign(df1['trend']) != np.sign(df2['trend'])) & (df1['trend'] != 0) & (df2['trend'] != 0))
    inconsistant_scores = sum(inconsistent_condition)

    # import ipdb;
    # ipdb.set_trace()

    df_base = df1[condition]
    df_condition1 = df1[condition1]
    df_condition2 = df2[condition2]

    consistent_df = pd.concat([df_base, df_condition1, df_condition2])

    # consistent results
    lwin = sum(consistent_df[f"{model1}_score"] > consistent_df[f"{model2}_score"])
    tie = sum(consistent_df[f"{model1}_score"] == consistent_df[f"{model2}_score"])
    rwin = sum(consistent_df[f"{model1}_score"] < consistent_df[f"{model2}_score"])

    assert len(df1) == lwin + tie + rwin + inconsistant_scores

    tie = tie + inconsistant_scores

    return lwin, tie, rwin


def convert_to_percent(lwin, tie, rwin):
    total = lwin + tie + rwin

    p_lwin = int(lwin * 100 / total)
    p_tie = int(tie * 100 / total)
    p_rwin = int(rwin * 100 / total)

    return p_lwin, p_tie, p_rwin


def get_gpt4_verdict(args):
    model1 = args.model1
    model2 = args.model2

    results = {}

    if args.cross:
        model1_vs_model2_comp_file, _ = get_comparison_out_file_path(args.model1, args.model2, args.gpt4_prompt_type,
                                                                     args.task, "cross")
        model2_vs_model1_comp_file, _ = get_comparison_out_file_path(args.model2, args.model1, args.gpt4_prompt_type,
                                                                     args.task, "cross")

        df1 = pd.read_json(model1_vs_model2_comp_file, lines=True)
        df2 = pd.read_json(model2_vs_model1_comp_file, lines=True)

        lwin, tie, rwin = get_winning_rate_consistent(df1, df2, model1, model2)

        p_lwin, p_tie, p_rwin = convert_to_percent(lwin, tie, rwin)

        results['cross'] = [p_lwin, p_tie, p_rwin]
        print(
            f"for cross-lingual: {model1} vs {model2} win rate is: \n {model1}: {p_lwin} \t tie: {p_tie} \t {model2}:{p_rwin}")
    else:
        for lang in [AR, EN]:
            model1_vs_model2_comp_file, _ = get_comparison_out_file_path(args.model1, args.model2,
                                                                         args.gpt4_prompt_type,
                                                                         args.task, lang)
            model2_vs_model1_comp_file, _ = get_comparison_out_file_path(args.model2, args.model1,
                                                                         args.gpt4_prompt_type,
                                                                         args.task, lang)

            df1 = pd.read_json(model1_vs_model2_comp_file, lines=True)
            df2 = pd.read_json(model2_vs_model1_comp_file, lines=True)

            lwin, tie, rwin = get_winning_rate_consistent(df1, df2, model1, model2)

            p_lwin, p_tie, p_rwin = convert_to_percent(lwin, tie, rwin)

            results[f'{code_to_language(lang)}'] = [p_lwin, p_tie, p_rwin]
            print(
                f"for {lang}: {model1} vs {model2} win rate is: \n {model1}: {p_lwin} \t tie: {p_tie} \t {model2}:{p_rwin}")
    return results


def generate_model_comparisons(args):
    results = get_gpt4_verdict(args)
    survey(args, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default='')
    parser.add_argument("--model2", type=str, default='')
    parser.add_argument("--fig_name_model1", type=str, help='name to be displayed in figure for model1')
    parser.add_argument("--fig_name_model2", type=str, help='name to be displayed in figure for model2')
    parser.add_argument("--gpt4_prompt_type", type=str, choices=GPT4_PROMPT_TYPE,
                        help="values can be generic, safety, summary", )
    parser.add_argument("--task", type=str, choices=TASKS, help="values can be vicuna, seeds, safety, summary", )
    parser.add_argument("--cross", action="store_true")

    args = parser.parse_args()

    generate_model_comparisons(args)
