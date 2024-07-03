import os, sys
import glob
import json
import numpy as np
# sys.path.append("..")
from tasks import base_tasks_en, base_tasks_ar, open_llm_ar_tasks, open_llm_en_tasks, math_tasks, long_context_tasks, \
    uae_tasks, base_tasks_hi
from batch_run import platform_results_path, get_all_model_names
import csv

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob

model_names = get_all_model_names()


def save_heatmap(data, xticklabels, yticklabels, title, sub_title, fig_path):
    # sns.heatmap(data, annot=True, cmap='viridis', fmt='.1f')
    # sns.heatmap(data, annot=False, cmap='YlGnBu', xticklabels=xticklabels, yticklabels=yticklabels)

    num_rows, num_columns = len(data),len(data[0])

    if num_rows>18: # till 8k context length we can have square boxes but as the rows become larger we want to adjust
                    # the figure size.
        fig_width = num_columns * 0.5
        fig_height = num_rows * 0.3
        plt.figure(figsize=(fig_width, fig_height))

    plt.clf()

    plt.clf()
    heatmap = sns.heatmap(data, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=xticklabels,
                          yticklabels=yticklabels, linewidth=.5)
    # plt.show()
    plt.xlabel('Depth')
    plt.ylabel('Context Length')
    heatmap.xaxis.set_label_position('top')
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    # plt.tick_params(axis='x', which='both', left=False, right=True, labelleft=False, labelright=True)

    plt.suptitle(title, y=1.02, fontsize=16)

    # Add a subtitle
    plt.title(sub_title, fontsize=14)
    plt.tight_layout()

    plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)


def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def read_all_related_needle_in_haystack_jsons(results_dir, model, task):
    file_list = glob.glob(results_dir + f"/{model}*{task}*.json")
    if task == "needle_in_haystack":
        file_list = [fl for fl in file_list if "needle_in_haystack_ar" not in fl]
    if len(file_list) == 0: return None

    consolidated_data = None
    print(f"For {model} and {task}, found following json files:\n{file_list}")
    for fl in file_list:
        data = read_json(fl)
        if consolidated_data is None:
            consolidated_data = data
        else:
            consolidated_data['examples'][task] += data['examples'][task]

    return consolidated_data


def generate_needle_in_haystack_plot(model, task, needle_results_json_file, title, subtitle):
    fig_file_path = needle_results_json_file[:-5] + ".png"
    if os.path.exists(fig_file_path):
        print(f"{fig_file_path}  exists; not generating needle heatmap again.")
        return

    results_dir = os.path.dirname(needle_results_json_file)
    data = read_all_related_needle_in_haystack_jsons(results_dir, model, task)
    if data is None:
        return

    # data = read_json(needle_results_json_file)
    cont_2dep_dict = {}
    needle_key = list(data['examples'].keys())[0]  # either needle_in_haystack_ar or needle_in_haystack
    for d in data['examples'][needle_key]:
        cont_len = int(d['context_len'])
        depth = int(d['depth'])
        score = int(d['score'])

        curr_cont_len_dict = cont_2dep_dict.get(cont_len, {})

        scores_list = curr_cont_len_dict.get(depth, [])
        scores_list.append(score)
        curr_cont_len_dict[depth] = scores_list

        cont_2dep_dict[cont_len] = curr_cont_len_dict

    for cont_len in cont_2dep_dict.keys():
        curr_cont_len_dict = cont_2dep_dict[cont_len]
        for depth in curr_cont_len_dict.keys():
            # curr_cont_len_dict[depth] = sum(curr_cont_len_dict[depth])//len(curr_cont_len_dict[depth])
            curr_cont_len_dict[depth] = max(curr_cont_len_dict[depth])
        cont_2dep_dict[cont_len] = curr_cont_len_dict

    sorted_cont_2dep_dict = dict(sorted(cont_2dep_dict.items()))
    for k in sorted_cont_2dep_dict.keys():
        ddict = dict(sorted(sorted_cont_2dep_dict[k].items()))
        sorted_cont_2dep_dict[k] = ddict

    csv_data = [["    "] + list(sorted_cont_2dep_dict[k].keys())]

    heatmap_data = []
    xticklabels = list(sorted_cont_2dep_dict[k].keys())
    yticklabels = []

    for cont_len in sorted_cont_2dep_dict.keys():
        data_entry = [cont_len]
        yticklabels.append(cont_len)
        _d = []
        for depth in sorted_cont_2dep_dict[cont_len].keys():
            data_entry.append(sorted_cont_2dep_dict[cont_len][depth])
            _d.append(sorted_cont_2dep_dict[cont_len][depth])
        csv_data.append(data_entry)
        heatmap_data.append(_d)

    # for d in csv_data:
    #     print(" ".join([str(jj) for jj in d]))

    # csv_file_path = needle_results_json_file[:-5]+".csv"
    # print(csv_file_path)
    # write_csv(csv_file_path,csv_data)

    save_heatmap(heatmap_data, xticklabels, yticklabels, title, subtitle, fig_file_path)


def write_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)


def get_score(result_file, task):
    try:
        results = read_json(result_file)
    except:
        print(result_file)
        return -1
    try:
        if task in {'truthfulqa'}:
            return results['results'][f'{task}_mc']['mc2'] * 100
        if task in ['truthfulqa_mc_ar','truthfulqa_hi']:
            return results['results'][f'{task}']['mc2'] * 100
        if task in ['mmlu', 'mmlu_ar', 'mmlu_hu_ar','mmlu_hi']:
            l1 = list(map(lambda x: x['acc_norm'], results['results'].values()))
            return np.mean(l1) * 100
            # corr_preds = list(map(lambda x: x['acc_norm'] * x['num_samples'], results['results'].values()))
            # num_samples = list(map(lambda x: x['num_samples'], results['results'].values()))
            # avg_acc = round(sum(corr_preds) * 100 / sum(num_samples), 2)
            # return avg_acc

        if task in {'crowspairs', 'crowspairs_ar'}:
            return np.mean(list(map(lambda x: x['pct_stereotype'], results['results'].values()))) * 100
        if task in {'arc_easy', 'openbookqa', 'piqa', 'mathqa', 'siqa', \
                    'exams_ar', 'digitised_ar', 'hellaswag_ar', 'piqa_ar', 'siqa_ar', 'arc_challenge_ar', \
                    'openbookqa_ar', "agqa", "agrc", "uae_ar", "uae_en", "tpo", "quality", "ardc_long_context"}:
            return results['results'][task]['acc_norm'] * 100
        if task in ['arc_challenge','arc_hi', 'hellaswag','hellaswag_hi', "gpqa_diamond_zeroshot", "agieval_en"]:
            try:
                sc = results['results'][task]['acc_norm']
            except KeyError:
                sc = results['results'][task]['acc_norm,none']
            return sc * 100
        if task in ["arc_challenge_math", 'hellaswag_math']:
            t = 'arc_challenge' if task == 'arc_challenge_math' else 'hellaswag'
            sc = results['results'][t]['acc_norm,none']
            return sc * 100
        if task == 'mmlu_stem':
            return results['results'][task]['acc,none'] * 100
        if task in ['race', 'winogrande', 'boolq', 'triviaqa', 'boolq_ar']:
            return results['results'][task]['acc'] * 100
        if task in ['gsm8k']:
            return results['results'][task]['exact_match,flexible-extract'] * 100
        if task in ['summ_en', 'summ_ar']:
            scores = []
            for eval in ['rouge1', 'rouge2', 'rougeLsum']:
                ress = results['results'][task][eval] * 100
                scores.append(ress)
            scores.append(results['results'][task]["bleu"])
            return scores
        if task == 'safety_helpful':
            scores = []
            _tasks = base_tasks_ar[task][0].split(",")
            for t in _tasks:
                try:
                    ress = results['results'][t]['acc'] * 100
                except KeyError:
                    print(f"for {t} there is no acc_norm evaluations.")
                    sys.exit()
                scores.append(ress)
            return scores
        if task in ["iwslt17-en-ar", "iwslt17-ar-en"]:
            return results['results'][task]["bleu"]
        if task in ['xtreme_ar_en', 'xtreme_en_ar']:
            return round(results['results'][task]['score'], 2)
        if task == 'math':
            corr_preds = list(map(lambda x: x['acc'] * x['num_samples'], results['results'].values()))
            num_samples = list(map(lambda x: x['num_samples'], results['results'].values()))
            avg_acc = round(sum(corr_preds) * 100 / sum(num_samples), 2)

            all_accs = list(map(lambda x: round(x['acc'] * 100, 2), results['results'].values()))
            all_accs.append(avg_acc)
            return all_accs
        if task == "drop":
            return results['results'][task]['f1,none'] * 100
        if task == 'bbh':
            tasks = ["bbh_zeroshot_penguins_in_a_table", "bbh_zeroshot_boolean_expressions",
                     "bbh_zeroshot_multistep_arithmetic_two"]
            scores = []
            num_samples = [146, 250, 250]
            for t in tasks:
                scores.append(results['results'][t]['exact_match,flexible-extract'])
            return round(np.sum(np.array(scores) * np.array(num_samples)) / sum(num_samples), 3) * 100
        else:
            print(f"Error: Key not found: {task}")
            return None
    except KeyError:
        print(f"\n\nFor {task}: key not found.\n\n")
        return -1


def append_scores_to(table, score):
    if score and isinstance(score, list):
        for _score in score:
            table.append(f"{_score:.1f}")
    elif score:
        table.append(f"{score:.1f}")
    else:
        table.append("")
    return table


def get_score_subset_mt(results_hu, results_mt):
    mt_score = []
    for subject in list(results_hu['versions'].keys()):
        subject = subject.replace('hu-', '')
        subject = subject.replace('hu_', '')
        mt_score.append(results_mt['results'][subject]['acc_norm'])
    return np.mean(mt_score) * 100


math_eval_tasks = ['math_algebra', 'math_counting_and_prob', 'math_geometry', 'math_intermediate_algebra',
                   'math_num_theory', 'math_prealgebra', 'math_precalc', 'math_asdiv', 'math']


def create_results_header(task_list):
    results_head = ['model_name']
    for t in task_list:
        if 'needle_in_haystack' in t:
            continue
        if t == 'math':
            results_head += math_eval_tasks
            continue
        results_head.append(t)
    # task_list = [t for t in task_list if 'needle_in_haystack' not in t]
    # results_head = ['model_name'] + task_list
    print(', '.join(results_head))
    return results_head


def update_results_table(table, model, task, task_details, task_type):
    _, num_shot, _ = task_details
    result_file = os.path.join(platform_results_path, f"output_{num_shot}shot", f"{model}_{task}.json")
    if "needle_in_haystack" in task:
        subtitle = "English Long Context" if task == "needle_in_haystack" else "Arabic Long Context"
        generate_needle_in_haystack_plot(model, task, result_file, title=model, subtitle=subtitle)
    if os.path.exists(result_file):
        score = get_score(result_file, task)
        table = append_scores_to(table, score)
    else:
        if task == "math":
            for i in range(len(math_eval_tasks)):
                table.append("")
        else:
            table.append("")
    if task == 'mmlu_hu_ar':
        mmlu_mt_fpath = os.path.join(platform_results_path, f"output_{num_shot}shot", f"{model}_mmlu_ar.json")
        if os.path.exists(result_file) and os.path.exists(mmlu_mt_fpath):
            results_hu = json.load(open(result_file))
            results_mt = json.load(open(mmlu_mt_fpath))
            score = get_score_subset_mt(results_hu, results_mt)
            table = append_scores_to(table, score)
        else:
            table.append("")
    return table

def is_result(table):
    for t in table[1:]: #first element is model name so ignoring that
        if t != '':
            return True
    return False

def generate_results(task_dict, task_type):
    results_data = []
    print(f'\n------------{task_type}----------------')
    columns = create_results_header(task_dict.keys())
    if task_type == "Arabic":
        columns.insert(2, 'mmlu_mt_ar_SUB')
    # results_data.append(results_head)
    for model in model_names:  # Iter Models
        table = [model]
        for task, task_details in task_dict.items():
            table = update_results_table(table, model, task, task_details, task_type)
        print(", ".join(table))
        if is_result(table): # to avoid adding rows where result is not present for any task
            results_data.append(table)
    try:
        results_df = pd.DataFrame(results_data, columns=columns)

    except ValueError as e:
        print(e)
        print(f"Error while converting dataframe for {task_type}")
        num_columns = len(columns)
        for r in results_data:
            if num_columns != len(r):
                print(f"Probable cause is: Number of columns: {columns} is not equal to Number of results: {len(r)}")
                print(r)
                break
        print(f"Unable to create dataframe, the results for {task_type} will not be written.")
        return None
    return results_df

def generate_results_excel(task_dict_and_task_type_tuples):
    result_dfs = []
    for (task_dict,task_type) in task_dict_and_task_type_tuples:
        current_results_df = generate_results(task_dict, task_type)
        result_dfs.append((current_results_df,task_type))

    with pd.ExcelWriter('lm_harness_results.xlsx', engine='xlsxwriter') as writer:
        for (result_df,task_type) in result_dfs:
            if result_df is not None:
                result_df.to_excel(writer, sheet_name=task_type, index=False)




task_dict_and_task_type_tuples = [
    #(task_dict_from_tasks.py_file,task_type_name_for_the_corresponding_xlsheet)
    (base_tasks_en, "English"),
    (base_tasks_ar, "Arabic"),
    (open_llm_en_tasks, "Few shot English"),
    (open_llm_ar_tasks, "Few shot Arabic"),
    (math_tasks, "Maths"),
    (long_context_tasks, "Long Context"),
    (uae_tasks, "UAE"),
    (base_tasks_hi,"Hindi"),
]
generate_results_excel(task_dict_and_task_type_tuples)