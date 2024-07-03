#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=80g
#SBATCH --time=14-00:00:00
#SBATCH --output=./slurm/%j.out
##SBATCH --exclude=p4-r76-a.g42cloud.net,p4-r80-a.g42cloud.net
##SBATCH --array=0-1


model_name=$1
model_path=$2
task=$3
gpt4_prompt_type=$4

BASEDIR=$(dirname $0)


nvidia-smi
free -h
uname -a
hostname
echo "JOB id: $SLURM_JOB_ID"
echo "JOB Array id: $SLURM_ARRAY_JOB_ID"

date


python ${BASEDIR}/model_text_gen.py --model_name $model_name --model_path $model_path --task $task --lang en


python3 ${BASEDIR}/judge_single_using_gpt.py --model $model_name --gpt4_prompt_type $gpt4_prompt_type --task $task --lang en


python ${BASEDIR}/calc_single_gpt_score.py --model $model_name --gpt4_prompt_type $gpt4_prompt_type --task $task


date

## sbatch eval_single_with_ref.sh jais_590M_v12p2_gbs256_hf_24690 /vast/core42-nlp/shared/onkar.pandit/jais/ft/jais_590M_v12p2_gbs256_hf_24690/hf_24690 vicuna
