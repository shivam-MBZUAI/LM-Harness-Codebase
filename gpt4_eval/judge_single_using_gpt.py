import os
import json
import argparse
from tqdm import tqdm
import pandas as pd

from utils import AR, EN, TASKS, VICUNA, SELF_INSTRUCT, SUMMARIZATION, SAFETY, IQEVAL, FRESHQA, ITC, QUANT, GPT4_PROMPT_TYPE
from utils import jsonl_out_path, xls_out_path, gpt4_gen, parse_score_single, read_json, read_jsonl, \
    write_jsonl, get_response_out_file_path, gpt4_prompts_file, get_single_out_file_path


ref_answer_keys = {
    # VICUNA: 'question_ar' if self.lang == AR else 'question',
    # SELF_INSTRUCT: 'question_ar' if self.lang == AR else 'question',
    # SUMMARIZATION: 'question_ar' if self.lang == AR else 'question',
    # SAFETY: "question",
    IQEVAL: 'answerkey',
    FRESHQA: 'answer_0',
    ITC: 'gold_answer',
    QUANT: 'gold_answer',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name", default='jais-ph3-7p2')
    parser.add_argument("--model_response", type=str, help="response filepath for model in jsonl format.")
    parser.add_argument("--gpt4_prompt_type", type=str,
                        help="values can be generic, single-with-ref, single-with-ref-and-doc", default='single-with-ref')
    parser.add_argument("--task", type=str, choices=TASKS, help="values can be freshqa, vicuna, seeds",
                        default='vicuna')
    parser.add_argument("--lang", type=str, choices=[AR, EN], help="language en or ar", default="en")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    prompts = read_json(gpt4_prompts_file)
    single_out_jsonl_path, single_out_xls_path = get_single_out_file_path(
        args.model,
        args.gpt4_prompt_type,
        args.task,
        args.lang
        )

    model_response_path = args.model_response if args.model_response else get_response_out_file_path(args.model,
                                                                                                     args.task,
                                                                                                     args.lang)

    data = read_jsonl(model_response_path)

    if os.path.exists(single_out_jsonl_path):  # RERUN: fix GPT-4 no response Errors
        with open(single_out_jsonl_path) as f:
            old_data = [json.loads(line) for line in f]
    else:
        old_data = [dict()] * len(data)

    output = []
    for line, oline in tqdm(zip(data, old_data)):
        # GPT-4 evaluation already there  AND scores can be extracted
        if ('GPT4-review' in oline and oline['GPT4-review'] != 'error') and (oline[f'{args.model}_score'] != -1):
            output.append(oline)
            continue

        ref_answer_key = ref_answer_keys[args.task]

        # Get GPT-4 Evaluation
        gpt4_eval_prompt = prompts[args.lang][args.gpt4_prompt_type]["prompt_template"].format(
            question=line['question'],
            ref_answer_1=line[ref_answer_key],
            answer=line['response'],
            prompt_end=prompts[args.lang][args.gpt4_prompt_type]["prompt_end"],
        )
        review = gpt4_gen(prompts[args.lang][args.gpt4_prompt_type]["system_prompt"], gpt4_eval_prompt, 2048)
        score = parse_score_single(review)

        oline = {
            'question'           : line["question"],
            'ref_answer_1'       : line[ref_answer_key],
            f'{args.model}'      : line["response"],
            'GPT4-review'        : review,
            f'{args.model}_score': score,
            'origin'             : line['origin'],
        }
        output.append(oline)

    # dump as jsonl
    print(f"Writing to jsonl file at: {single_out_jsonl_path}")
    write_jsonl(output, single_out_jsonl_path)

    # dump as excel
    try:
        df = pd.DataFrame(output)
        df.to_excel(single_out_xls_path, index=False)
    except:
        print(f"Error while creating xls file but jsonl output is already created: {single_out_jsonl_path}")


if __name__ == "__main__":
    main()
