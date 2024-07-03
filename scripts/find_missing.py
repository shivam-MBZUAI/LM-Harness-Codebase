import os
import argparse

from tasks import task_info_en, task_info_ar
task_info = dict(task_info_en)
task_info.update(task_info_ar)

tasks_en = dict(task_info_en).keys()
tasks_ar = dict(task_info_ar).keys()
# tasks = ['siqa_ar', 'exams_ar', 'truthfulqa', 'digitised_ar', 'arc_challenge', 'hellaswag', 'hellaswag_ar',
#          'arc_challenge_ar', 'winogrande', 'siqa', 'openbookqa_ar', 'truthfulqa_mc_ar', 'mmlu_hu_ar', 'crowspairs',
#          'mmlu_ar', 'piqa_ar', 'boolq', 'piqa', 'race', 'mmlu', 'openbookqa', 'boolq_ar', 'crowspairs_ar',"agqa","agrc",
#          'summ_ar','summ_en']


def find_missing_tasks(out_path, file_starts):
    json_files = [file for file in os.listdir(out_path) if file.endswith('.json') and file.startswith(file_starts)]
    current_tasks = [f.split(file_starts)[-1].split(".json")[0] for f in json_files]
    current_tasks = [f if not f.startswith("_") else f[1:] for f in current_tasks]
    missed_en = [t for t in tasks_en if t not in current_tasks]
    missed_ar = [t for t in tasks_ar if t not in current_tasks]
    print(f"for {file_starts}: out of total {len(tasks_en) + len(tasks_ar)} tasks {len(missed_en)+len(missed_ar)} tasks are missed: \n English: {missed_en} \n Arabic: {missed_ar}")
    return json_files

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path",
                        default="../output_0shot",
                        help="results path")

    parser.add_argument("--file_starts", default="")
    args = parser.parse_args()

    # out_path = "/nfs/users/ext_onkar.pandit/softwares/haonan_lm_eval/output_0shot"
    # file_starts = "m_30b_ft_v6_unpacked_hf_6908"
    # file_starts = "llama-30b_instruct"

    out_path = args.out_path
    file_starts = args.file_starts
    jf = find_missing_tasks(out_path, file_starts)