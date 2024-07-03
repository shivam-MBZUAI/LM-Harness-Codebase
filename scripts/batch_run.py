import os
import platform
import subprocess
import json
import sys

from tasks import base_tasks_en, base_tasks_ar, generation_gpt4_eval_tasks, open_llm_en_tasks,open_llm_ar_tasks, math_tasks, long_context_tasks, uae_tasks, base_tasks_hi

def read_text_file(filename):
    try:
        with open(filename, "r") as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        print("File not found.")
        return None




plat_id_conf_path = "/nfs_shared/lmeval.conf"
try:
    platform_id = read_text_file(plat_id_conf_path).strip()
except:
    platform_id = None
if not platform_id:
    platform_id = platform.uname().release # for v100

model_name_to_path = "./model_name_to_path.jsonl"

task_info = dict()
# task_info.update(base_tasks_en)
# task_info.update(base_tasks_ar)
task_info.update(base_tasks_hi)
task_info.update(generation_gpt4_eval_tasks)
# task_info.update(open_llm_en_tasks)
# task_info.update(open_llm_ar_tasks)
# task_info.update(math_tasks)
task_info.update(long_context_tasks)
# task_info.update(uae_tasks)

taiga_a100_id = "5fd3cef7-510b-4efa-85b7-fbdfdb7ffd8f"
taiga_h100_id = "03e47ceb-6c5a-4934-9282-6c765a0b0e1d"
v100_id = '5.4.0-104-generic'
taiga_big_h100_id = '5.15.0-102-generic'

kernel_to_platform_args = {
    v100_id: {'platform_cmd': 'srun --nodes=1'
                                          ' --output={current_dir}/stdout/%j_{model_name}_{task}.log --partition=iiai --time=14-00:00:00'
                                          ' --ntasks-per-node=1 --cpus-per-task=6 --gres=gpu:{gpu} --mem={mem} --job-name={task}'
                                          ' --exclude=p4-r76-a.g42cloud.net,p4-r80-a.g42cloud.net ',
                          'model_path_key': 'v100_path', 'results_path':"/nfs/users/ext_sunil.kumar/nlp_data/LM_DATA/EvaluationResults/lm_harness_results"},  # v100
    '5.15.0-1035-oracle': {'platform_cmd': '', 'model_path_key': ''},  # ngc
    '5.15.0-1047-oracle': {'platform_cmd': 'srun --nodes=1'
                                          ' --output={current_dir}/stdout/%j_{model_name}_{task}.log --time=14-00:00:00'
                                          ' --ntasks-per-node=1 --cpus-per-task=6 --gres=gpu:{gpu} --mem={mem} --job-name={task}',
                          'model_path_key': 'lambda_a100_path','results_path':"/nfs/nlp_data/eval_results/lm_harness_results"}, #lambda A100
    taiga_a100_id:{'platform_cmd': 'srun --nodes=1'
                                          ' --output={current_dir}/stdout/%j_{model_name}_{task}.log --time=14-00:00:00'
                                          ' --exclude=g42-instance-03,g42-instance-01 --partition=gpu'
                                        #   ' --nodelist=g42-instance-[001-002]'
                                          ' --ntasks-per-node=1 --cpus-per-task=6 --gres=gpu:{gpu} --mem={mem} --job-name={task}',
                          'model_path_key': 'taiga_path','results_path':"/nfs_shared/NLP_DATA/LM_DATA/EvaluationResults/lm_harness_results"}, # taiga cluster
    taiga_h100_id:{'platform_cmd': 'srun --nodes=1'
                                          ' --output={current_dir}/stdout/%j_{model_name}_{task}.log --time=14-00:00:00'
                                          ' --exclude=g42-h100-001,g42-h100-002 --partition=core42users'
                                        #   ' --nodelist=g42-h100-[001-002]'
                                          ' --ntasks-per-node=1 --cpus-per-task=6 --gres=gpu:{gpu} --mem={mem} --job-name={task}',
                          'model_path_key': 'taiga_path','results_path':"/nfs_shared/NLP_DATA/LM_DATA/EvaluationResults/lm_harness_results"}, # h100 cluster
    taiga_big_h100_id:{'platform_cmd': 'srun --nodes=1'
                                          ' --output={current_dir}/stdout/%j_{model_name}_{task}.log --time=14-00:00:00'
                                          ' --partition=nlp'
                                        #   ' --nodelist=g42-h100-[048-055]'
                                          ' --ntasks-per-node=1 --cpus-per-task=6 --gres=gpu:{gpu} --mem={mem} --job-name={task}',
                          'model_path_key': 'big_h100_path','results_path':"/vast/core42-nlp/shared/NLP_DATA/LM_DATA/EvaluationResults/lm_harness_results"}, # big h100 cluster
}
# --nodelist=g42-h100-[055-070]
platform_results_path = kernel_to_platform_args[platform_id]['results_path']

lm_harness_command = 'python {work_dir}/main.py --model {lm_model_class} --model_args pretrained={model_path},use_accelerate=True --tasks {task_map} --num_fewshot {num_fewshot} --output_path {out_file}  --device cuda {task_args}'
lm_harness_command_vllm = 'python {work_dir}/main.py --model vllm --model_args pretrained={model_path},dtype={dtype},tensor_parallel_size={tensor_parallel_size},trust_remote_code=True,gpu_memory_utilization=0.95 --tasks {task_map} --num_fewshot {num_fewshot} --output_path {out_file}  --device cuda {task_args}'
gpt4_evals_comparison_command = 'bash {gpt4_eval_dir}/run_model_comparison_tasks.sh {model_name} {model_path} {model_to_compare_name} {model_to_compare_path} {task_map} {num_fewshot}'

gpt4_single_eval_command = 'bash {gpt4_eval_dir}/eval_single_with_ref.sh {model_name} {model_path} {task_map} {num_fewshot}'

# new_lm_harness_command = "bash;conda init;conda deactivate;conda activate /nfs_users/users/onkar.pandit/miniconda3/envs/new_lm;lm_eval --model hf \
#     --tasks {task_map} \
#     --num_fewshot {num_fewshot} \
#     --model_args pretrained={model_path},trust_remote_code=True,parallelize=True \
#     --batch_size 4 --output_path {out_file}"

# using hf
new_lm_harness_command = "lm_eval --model hf \
    --tasks {task_map} \
    --num_fewshot {num_fewshot} \
    --model_args pretrained={model_path},trust_remote_code=True,parallelize=True,dtype=bfloat16 \
    --batch_size 4 --output_path {out_file}"

# new_lm_harness_command = "lm_eval --model vllm \
#     --tasks {task_map} \
#     --num_fewshot {num_fewshot} \
#     --model_args pretrained={model_path},trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=8 \
#     --batch_size auto --output_path {out_file}"

# new_lm_harness_command = "accelerate launch -m lm_eval --model hf \
#     --tasks {task_map} \
#     --num_fewshot {num_fewshot} \
#     --model_args pretrained={model_path},trust_remote_code=True \
#     --batch_size 4 --output_path {out_file}"

iq_test_command = 'bash {iq_test_dir}/run.sh {model_name} {model_path} {iq_ques_type} {prompt_type}'

current_dir = os.getcwd()
work_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
gpt4_eval_dir = os.path.join(work_dir, 'gpt4_eval')
iq_test_dir = os.path.join(work_dir, 'iq_test')
armmlu_dir = os.path.join(work_dir, 'ArabicMMLU')

armmlu_command = 'bash {armmlu_dir}/run.sh {model_name} {model_path}'

slurm_relative_path = f"stdout"
os.makedirs(os.path.join(current_dir,slurm_relative_path), exist_ok=True)

def read_jsonl(f_name):
    data = []
    with open(f_name) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_text_file(filename):
    try:
        with open(filename, "r") as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        print("File not found.")
        return None


def get_model_details_dict(model_name, model_name_to_path_dicts):
    for d in model_name_to_path_dicts:
        if d['name'] == model_name:
            return d
    print(f"{model_name} does not exist in the list; model path and details not found. Please add and re-run.")
    sys.exit()


def get_model_path(model_name_dict, model_path_key):
    if 'path' in model_name_dict.keys():
        return model_name_dict['path']
    else:
        print(model_name_dict)
        return model_name_dict[model_path_key]


def get_all_model_names():
    model_name_to_path_dicts = read_jsonl(model_name_to_path)
    model_names = [d['name'] for d in model_name_to_path_dicts]
    return model_names



def run():
    model_name_to_path_dicts = read_jsonl(model_name_to_path)

    # model_to_be_evaluated = ["llama2-7b","llama2-13b","llama2-70b-hf","llama3_8b_base","llama3-70b-base","llama2_13b_hi_adapted_base"]
    # model_to_be_evaluated = ["1p3b_v12p2_gbs256_hf_24690"]
    model_to_be_evaluated = ["llama3_8b_ar_adapted_v12p2_3epochs"]

    # model_to_be_evaluated = ["llama2-70b-hf"]

    # model_to_be_evaluated = ["m_13b_dpo_ultrachat_beta0.1","m_13b_dpo_ultrachat_beta0.2","m_13b_dpo_ultrachat_beta0.3","m_13b_dpo_ultrachat_beta0.4","m_13b_dpo_ultrachat_beta0.5"]

    # model_to_be_evaluated = ["30b_ph2_ft_7p1_cont_len_2k_hf_7449","30b_ph2_ft_7p1_cont_len_8k_hf_7565","30b_ph2_ft_9p5_cont_len_2k_hf_7460","30b_ph2_ft_v9p5_2k_incr_8k_hf_9986"]
    # model_to_be_evaluated = ["jais-13b", "jais-30b-v1", "jais-30b-v3"]
    # model_to_be_evaluated = ["30b_ph2_ft_v7p1_2k_incr_8k_hf_9970", "30b_ph1_ft_v7p1_2k_incr_8k_hf_9932"]

    # model_to_be_evaluated = ["jais-30b-v1-chat","30b_ph2_ft_7p1_cont_len_8k_hf_7565","30b_ph2_ft_v9p5_2k_incr_8k_hf_9986","jais-13b-chat"]
    # model_to_be_evaluated = ["30b_ph1_ft_v7p1_8k_hf_2690", "30b_ph1_ft_v7p1_8k_hf_8070"]
    # model_to_be_evaluated = ["30b_ph2_ft_v7p1_2k_hf_7449","rs_sft_30b_v7p1_2k_clen_3R_nectar_jaisdb_hf_151"]
    # model_to_be_evaluated = ["dpo_rs_best_random_data_30b_ph3_ft_7p1_8k_hf_7565_cpt_500"]
    # model_to_be_evaluated = ['30b_ph2_ft_7p2_8k_hf_7482']
    # model_to_compare_name = "llama_7b_lr_3e-4_shorter_schedule_mix_1-1_ft_v7p2_4k_hf_8405"  # name of the model which will be compared in gpt4 evaluations
    # model_to_be_evaluated = ["m_13b_ft7p3_2k_cpt_8196","jais-13b-chat","13b_ph1_ft_7p1_2k_hf_7449" ]
    # model_to_be_evaluated = ["ft_6.7b_ft_bs3072_hf_14812", "llama_7b_lr1.5e-4_mix_1-1_ft_v7p2_4k_hf_8405", "llama_7b_lr_3e-4_shorter_schedule_mix_1-1_ft_v7p2_4k_hf_8405"]        # ["llama2-7b"]
    # model_to_be_evaluated = ["llama2-13b-chat","jais-13b-chat"]

    # model_to_be_evaluated = ["llama2-13b-adapted_v7p2_4k_hf_8405","llama2-13b-chat","codellama-13b-instruct-hf"]


    # model_to_be_evaluated = ["llama2_70b_chat_ift_v10_math_iq_hf_4860", "llama2_70b_chat_ift_v10p1_math_iq_hf_8346",
    #                          "llama2-70b-chat-hf"]
    # model_to_be_evaluated = ["llama3-70b-base","llama3-70b-chat"]
    #
    # model_to_be_evaluated = ["13B_ft_7p6_2k_hf_8598", "30B_ph3_ift_v7p6_hf_8432"]      #"30b_ph3_ft_7p2_8k_hf_7257"]


    # model_to_be_evaluated = ['jais-13b-chat','jais-30b-v1-chat','30b_ph2_ft_7p1_cont_len_8k_hf_7565','30b_ph3_ft_7p2_8k_hf_7257']
    # model_to_be_evaluated = ['llama2-13b-adapted_v7p2_4k_hf_8405_maths_cpot_hf_1839']
    # model_to_be_evaluated = ["gemma-2b","gemma-2b-it","gemma-7b","gemma-7b-it","mistral-7B-v0.1","mistral-7B-Instruct-v0.2"]
    # model_to_be_evaluated = ['llama3_8b_instruct']
    # model_to_be_evaluated = [
    # "590m_Arav5_hf_209422",
    # "jais_590M_v12p2_gbs256_hf_24690",
    # "1p3B_Arav5_hf_187942",
    # "1p3b_v12p2_gbs256_hf_24690",
    # "2p7B_v3_hf_162883",
    # "2p7b_v12p2_gbs256_hf_24690",
    # "6p7B_Arav5_hf_143721",
    # "6p7b_v12p2_gbs512_hf_12540",
    # "13B_Arav5_hf_122162",
    # "13b_v12p2_gbs512_hf_12345",
    # "30b_ph3_pt_hf_260648",
    # "30B_3epochs_ift_v12p2_16k_hf_11342",
    # "30B_3epochs_ift_v12p2_8k_hf_12729",
    # "llama2_7b_ex100p_subwordmean_hf_10060",
    # "llama2_7B_ift_v12p2_4k_hf_13175",
    # "llama2-13b-adapted_70938",
    # "llama2_13B_ift_v12p2_4k_hf_13175",
    # "llama2-70b-pt_hf_94316",
    # "llama2_70b_ar_ift_v12p2_hf_13175",
    # ]
    # model_to_be_evaluated = ["llama2-70b-hf","llama2-70b-chat-hf"]
    # model_to_be_evaluated = ["llama2_70b_QCC_PP_ft_v10p1_hf_9180"]

    # model_to_be_evaluated = ["llama2-13b-ar_base_4k_maths_cpot_hf_1839","llama2-13b_base_4k_maths_cpot_hf_1839"]
    # model_to_be_evaluated = ['llama2-13b_ar_base_4k_maths_cpot_openmath_orca_hf_7548']
    # model_to_be_evaluated = ["Yi-34b","Yi-34b-chat","Yi-6b","Yi-6b-chat"]
    # model_to_be_evaluated = ["llama2-70b-pt_hf_39500","llama2-70b-pt_offramp_hf_59205"]
    # model_to_be_evaluated = ['mistral-7b-instruct_maths_cpot_openmath_orca']
    # model_to_be_evaluated = ['llama2-70b-ft_v7p4_math_hf_10227']
    # model_to_be_evaluated = ["llama2_70b_base_ift_v10_math_iq_hf_7517","llama2_70b_chat_ift_v10_math_iq_hf_7289","llemma_34b_ift_v10_math_iq_hf_7517"]
    # model_to_be_evaluated = ["llama2_70b_chat_math_ift_v10p1"]   # ['30B_ph3_16K_pt_hf_283648']           # ['30b_ph3_ft_7p4_math_hf_10111']

    # model_to_be_evaluated = ["llama2_70b_chat_ift_v10_math_iq_hf_4860", "llama2_70b_chat_ift_v10p1_math_iq_hf_8346",
    #                          "llama2-70b-chat-hf"]
    # model_to_be_evaluated = ['30b_ph3_pt_long_cont_16k_hf_283648']
    # model_to_be_evaluated = ["llama2-13b-ar_base_4k_maths_cpot_hf_1839","llama2-13b_base_4k_maths_cpot_hf_1839"]

    # model_to_be_evaluated = ["falcon_180_base", "llama3-70b-base"]

    # model_to_be_evaluated = ["llama2-13b-ar_base_4k_maths_cpot_hf_1839","llama2-13b_base_4k_maths_cpot_hf_1839"]
    # model_to_be_evaluated = ["13B_ftv7p6_2k_hf_12213"]

    # model_to_be_evaluated = ["llama2-13b-chat_maths_iq_reasoning_data_hf_1070","llama2-70b-chat_maths_iq_reasoning_data_hf_1070","llemma-34b_ft_maths_iq_reasoning_data_hf_1091"]

    # model_to_be_evaluated = ["llama2-13b-chat", "llama2-70b-chat-hf"]
    # model_to_be_evaluated = ["Jamba-v0.1"]
    # model_to_be_evaluated = ["llama2_70b_base_ift_v10_math_iq_hf_2506"]
    # model_to_be_evaluated = ['llama2-70b-ft_v7p4_math_hf_10227']

    ## FOR MATHS eval2
    # model_to_be_evaluated = ["llama3-70b-ft_v10p2_hf_7845", "falcon_ft_v10p2_hf_8072", "llama3-70b-chat", "llama3-70b-ft_v10p1_hf_7323", "falcon_ft_v10p1_hf_4305", "falcon_180_instruct"]

    # model_to_be_evaluated = ['llama2-70b-ft_v7p4_math_hf_10227']
    # model_to_be_evaluated = ["llama3-70b-ft_v10p2_hf_7845","falcon_ft_v10p2_hf_8072"]
    # model_to_be_evaluated = ['llama2_70b_v2_ift_v11_hf_11992']
    # model_to_be_evaluated = ["llama2-70b-ft_v7p4_math_hf_10227", "jamba", "llama3-70b-chat","30b_ph3_ft_7p2_8k_hf_7257"]

    # model_to_be_evaluated = ["llama2_70B_arabic_offramped_ift_math_iq_4k"]

    # model_to_be_evaluated = ["meta_math_mistral-7b","meta_math_llema-7b", "wizard_math-13b-V1.0","meta_math-7b-v1.0","wizard_math-7b-V1.0"]
    # model_to_be_evaluated = ["llama2-13b-adapted"]
    # model_to_be_evaluated = ["llama_subword_mean_extend_100p_lr_1.5e-4",
    #                          "llama_extend_100p_lr1.5e-4_mix_1-1_en-ar",
    #                          "llama_extend_100p_lr1.5e-4_mix_1-3_en-ar",
    #                          "llama_extend_100p_lr1.5e-4_mix_1-9_en-ar",
    #                          "llama_extend_100p_lr3e-4_mix_1-1_en-ar",]
                             # "llama2-7b","llama2-13b"]

    # model_to_be_evaluated = ["13B_ft_7p6_2k_hf_8598"]  # ["30B_ph3_ift_v7p6_hf_8432"]#   #,       #"30b_ph3_ft_7p2_8k_hf_7257"]
    # model_to_compare_name = "llama_7b_lr1.5e-4_mix_1-1_ft_v7p2_4k_hf_8405"        # "llama_7b_lr1.5e-4_mix_1-1_ft_v7p2_4k_hf_8405"  # name of the model which will be compared in gpt4 evaluations

    # model_to_be_evaluated = ["acegpt-13b-v1.5-chat"]  #, "acegpt-13b-v1.5"]

    # model_to_compare_name = "30b_ph3_ft_7p2_8k_hf_7257"
    # model_to_compare_name = 'jais-13b-chat'
    # model_to_compare_name = 'llama3-70b-chat'
    # model_to_compare_name = '30B_ph3_ift_v7p6_hf_8432'
    model_to_compare_name = 'llama3_8b_instruct'
    model_to_compare_name = 'llama3_pro_10b_ift_v12p2'

    # model_to_compare_name = "m_13b_ft7p3_2k_cpt_8196"

    # 13b v7p6 comparsion
    # 13b_ph1_ft_7p1_2k_hf_7449 jais-13b-chat

    model_to_compare_path = get_model_path(get_model_details_dict(model_to_compare_name, model_name_to_path_dicts),
                                           kernel_to_platform_args[platform_id][
                                               'model_path_key'])  # path of the model which will be compared in gpt4 evaluations

    # tasks_to_be_run = ['vicuna', 'safety_gen']
    # tasks_to_be_run = task_info.keys()
    # tasks_to_be_run = task_info.keys()
    # tasks_to_be_run = uae_tasks.keys()
    # tasks_to_be_run = math_tasks.keys()
    # tasks_to_be_run = ['bbh','drop',"gpqa_diamond_zeroshot"]
    tasks_to_be_run = ['vicuna']  #, 'seeds']
    # tasks_to_be_run = open_llm_en_tasks.keys()
    # tasks_to_be_run = open_llm_ar_tasks.keys()
    # tasks_to_be_run = math_tasks.keys()
    # tasks_to_be_run = ['agieval_en', 'drop', "math"]
    # tasks_to_be_run = ['vicuna']  #, 'seeds']
    # tasks_to_be_run = ['iq_test']
    # tasks_to_be_run = ['race', 'arc_30b_ph2_ft_v7p1_2k_hf_7449_school_hackpchallenge', 'openbookqa', 'truthfulqa']
    # tasks_to_be_run = ['needle_in_haystack','needle_in_haystack_ar']
    # tasks_to_be_run = long_context_tasks.keys()
    # tasks_to_be_run = ['needle_in_haystack_ar']
    # tasks_to_be_run = ['piqa']
    # tasks_to_be_run = ['ardc_long_context', 'tpo']
    # tasks_to_be_run = ['xtreme_ar_en','xtreme_en_ar']

    # tasks_to_be_run = ['math','lila']


    gpt4_eval_tasks = generation_gpt4_eval_tasks.keys()  # ['summary','vicuna','seeds','safety_gen']


    for model_name in model_to_be_evaluated:
        for task in tasks_to_be_run:  # Tasks to be evaluated
            model_name_dict = get_model_details_dict(model_name, model_name_to_path_dicts)
            task_map, num_fewshot, task_args = task_info[task]

            task_args = "" if task_args=="-" else task_args


            if task in gpt4_eval_tasks:
                if task not in ['iqeval', 'itc', 'quant']:
                    task_cmd = gpt4_evals_comparison_command.format(
                        gpt4_eval_dir=gpt4_eval_dir,
                        model_name=model_name,
                        model_path=get_model_path(model_name_dict,
                                                kernel_to_platform_args[platform_id]['model_path_key']),
                        model_to_compare_name=model_to_compare_name,
                        model_to_compare_path=model_to_compare_path,
                        num_fewshot=num_fewshot,
                        task_map=task_map,
                    )
                else:
                    task_cmd = gpt4_single_eval_command.format(
                        gpt4_eval_dir=gpt4_eval_dir,
                        model_name=model_name,
                        model_path=get_model_path(model_name_dict,
                                                kernel_to_platform_args[platform_id]['model_path_key']),
                        num_fewshot=num_fewshot,
                        task_map=task_map,
                    )

                num_gpu = max(int(model_name_dict['gpu']),int(get_model_details_dict(model_to_compare_name, model_name_to_path_dicts)['gpu']))
            else:
                lm_harness_results_path = os.path.join(platform_results_path, f"output_{num_fewshot}shot")
                os.makedirs(lm_harness_results_path, exist_ok=True)
                out_file = os.path.join(lm_harness_results_path, f'{model_name}_{task}.json')

                if os.path.exists(out_file):  # if result exist
                    continue

                num_gpu = int(model_name_dict['gpu'])

                # if task in ["bbh", "gpqa_diamond_zeroshot", "drop"]:
                if task in math_tasks.keys():
                    task_cmd = new_lm_harness_command.format(
                        model_path=get_model_path(model_name_dict,
                                                  kernel_to_platform_args[platform_id]['model_path_key']),
                        num_fewshot=num_fewshot,
                        out_file=out_file,
                        task_map=task_map,
                        task_args=task_args
                    )
                elif task == 'iq_test':
                    task_cmd = iq_test_command.format(
                        iq_test_dir=iq_test_dir,
                        model_name=model_name,
                        model_path=get_model_path(model_name_dict,
                                              kernel_to_platform_args[platform_id]['model_path_key']),
                        iq_ques_type='orig', #orig for giving
                        prompt_type='mod_template' # 'deploy_prompt'
                    )
                elif task == 'ArabicMMLU':
                    task_cmd = armmlu_command.format(
                        model_name=model_name,
                        model_path=get_model_path(model_name_dict,
                                              kernel_to_platform_args[platform_id]['model_path_key']),
                    )
                else:
                    task_cmd = lm_harness_command.format(
                        work_dir=work_dir,
                        lm_model_class='hf-causal-experimental',
                        model_path=get_model_path(model_name_dict,
                                                  kernel_to_platform_args[platform_id]['model_path_key']),
                        num_fewshot=num_fewshot,
                        out_file=out_file,
                        task_map=task_map,
                        task_args=task_args
                    )

            slurm_cmd = kernel_to_platform_args[platform_id]['platform_cmd'].format(
                current_dir=current_dir,
                gpu=num_gpu,
                mem=model_name_dict['mem'],
                task=task,
                model_name=model_name,
            )
            if "needle_in_haystack" in task:
                min_context_len = 500
                max_context_len = 8000
                context_len_step = 500
                max_elements_per_list = 5

                context_lengths = list(range(max_context_len, min_context_len - context_len_step, -context_len_step))

                sublists = [context_lengths[i:i + max_elements_per_list] for i in
                            range(0, len(context_lengths), max_elements_per_list)]

                min_max_values = [(sublist[-1], sublist[0]) for sublist in sublists]

                for cont_counter,(min_cont,max_cont) in enumerate(min_max_values):
                    needle_params = f"--needle_in_haystack_params min_context_len={min_cont},max_context_len={max_cont},context_len_step=500"

                    task_cmd = lm_harness_command_vllm.format(
                        work_dir=work_dir,
                        model_path=get_model_path(model_name_dict,
                                                  kernel_to_platform_args[platform_id]['model_path_key']),
                        num_fewshot=num_fewshot,
                        out_file=f"{out_file[:-5]}_{cont_counter+1}.json",
                        task_map=task_map,
                        task_args=task_args,
                        tensor_parallel_size=1,
                        dtype="bfloat16",
                    )
                    cmd = f'{slurm_cmd} {task_cmd} {needle_params} &'
                    print(f"Executing: {cmd} \n \n \n ")
                    result = subprocess.run(cmd, shell=True, text=True, )
            else:
                cmd = f'{slurm_cmd} {task_cmd} &'
                print(f"Executing: {cmd} \n \n \n ")
                result = subprocess.run(cmd, shell=True, text=True, )


if __name__ == '__main__':
    run()