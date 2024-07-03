#!/bin/bash
#SBATCH --partition=iiai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=350g
#SBATCH --time=14-00:00:00
#SBATCH --output=./stdout/slurm-%j.out
#SBATCH --exclude=p4-r76-a.g42cloud.net,p4-r80-a.g42cloud.net


model1_name=$1
model1_path=$2

model2_name=$3
model2_path=$4

task=$5
gpt4_prompt_type=$6

BASEDIR=$(dirname $0)

nvidia-smi
free -h
uname -a
hostname
echo "JOB id: $SLURM_JOB_ID"
echo "JOB Array id: $SLURM_ARRAY_JOB_ID"

date
#echo "Script location: ${BASEDIR}"
#model_name=jais-ph3-7p2
#input_file=/nfs_shared/NLP_DATA/LM_DATA/Benchmark_datasets/gsm8k_150.jsonl
#output_file=/nfs_shared/NLP_DATA/LM_DATA/EvaluationResults/gpt4_eval_results/response/${model_name}_gsm8k_150.jsonl
#python3 model_text_gen.py --model_name $model_name --model_path /nfs_shared/NLP_DATA/LM_DATA/models/30b/30b_ph3_ft_v7p2_8k/hf_7257 --input_file $input_file --output_file $output_file --task vicuna --lang en

python ${BASEDIR}/model_text_gen.py --model_name $model1_name --model_path $model1_path --task $task --lang ar --use_chat_template
python ${BASEDIR}/model_text_gen.py --model_name $model1_name --model_path $model1_path --task $task --lang en --use_chat_template

python ${BASEDIR}/model_text_gen.py --model_name $model2_name --model_path $model2_path --task $task --lang ar --use_chat_template
python ${BASEDIR}/model_text_gen.py --model_name $model2_name --model_path $model2_path --task $task --lang en --use_chat_template

python3 ${BASEDIR}/compare_models.py --model1 $model1_name --model2 $model2_name --gpt4_prompt_type $gpt4_prompt_type --task $task --lang en &
p1=$!
python3 ${BASEDIR}/compare_models.py --model1 $model2_name --model2 $model1_name --gpt4_prompt_type $gpt4_prompt_type --task $task --lang en &
p2=$!
python3 ${BASEDIR}/compare_models.py --model1 $model1_name --model2 $model2_name --gpt4_prompt_type $gpt4_prompt_type --task $task --lang ar &
p3=$!
python3 ${BASEDIR}/compare_models.py --model1 $model2_name --model2 $model1_name --gpt4_prompt_type $gpt4_prompt_type --task $task --lang ar &
p4=$!

wait $p1 $p2 $p3 $p4


python ${BASEDIR}/gpt4_eval_scores.py --model1 $model1_name --model2 $model2_name --gpt4_prompt_type $gpt4_prompt_type --task $task

date
