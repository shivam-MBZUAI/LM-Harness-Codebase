import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
# import time

from utils import TASKS, AR, EN, GPT4_PROMPT_TYPE
from utils import gpt4_comp_path, get_single_out_file_path, write_jsonl
from quant_itc_category_mapping import quant_chapter_gpt_map

def calc_avg_score(args, df):
    """Calculates average scores per category.

    return a dict of category to scores if category is
    present else dict of model name to average score.
    """
    avg_scores = []
    if args.task == 'quant':
        ## map chapter names to categories:
        df['categories'] = df['origin'].map(quant_chapter_gpt_map)
        avg_scores = df[[f"{args.model}_score", 'categories']].groupby('categories').mean()
        # avg_scores = avg_scores.values.tolist()
        avg_scores = avg_scores.groupby(avg_scores.index)[f"{args.model}_score"].sum().to_dict()
    else:
        avg_scores = df[f"{args.model}_score"].mean()

    result = {f'{args.model}_score': avg_scores}
    return result


def get_gpt4_verdict(args):
    results = {}
    model_score_file, _ = get_single_out_file_path(
        args.model,
        args.gpt4_prompt_type,
        args.task,
        args.lang
        )
    print(f'Reading model scores from: [{model_score_file}]')
    df = pd.read_json(model_score_file, lines=True, orient='records')

    results = calc_avg_score(args, df)
    result_path = os.path.join(gpt4_comp_path, f'{args.model}_{args.task}_{args.gpt4_prompt_type}.json')
    write_jsonl(results, result_path)
    print(f"Wrote {args.model} scores at [{result_path}]: \n{results}")
    return results


def plot_category_spider(args, df):
    """ TODO: plots the spider graph of scores per category.
    """
    fig = px.line_polar(df, r = 'score', theta = 'category', line_close = True, category_orders = {"category": CATEGORIES},
                    color = 'model', markers=True, color_discrete_map=model_to_color, title="MT-Bench-En")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--fig_name_model", type=str, help='name to be displayed in figure for model')
    parser.add_argument("--gpt4_prompt_type", type=str, choices=GPT4_PROMPT_TYPE,
                        help="values can be generic, safety, summary", )
    parser.add_argument("--task", type=str, choices=TASKS, help="values can be vicuna, seeds, safety, summary", )
    parser.add_argument("--lang", type=str, choices=[AR, EN], help="language en or ar", default="en")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_arguments()
    ## TODO: Verify task type for singles

    results = get_gpt4_verdict(args)
    # if args.task == 'quant':
    #     plot_category_spider(args, results)
