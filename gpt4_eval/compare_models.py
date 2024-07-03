import os
import json
import argparse
from tqdm import tqdm
import pandas as pd

from utils import AR, EN, TASKS, VICUNA, SELF_INSTRUCT, SUMMARIZATION, SAFETY, IQEVAL, FRESHQA, ITC, QUANT, GPT4_PROMPT_TYPE
from utils import get_response_out_file_path, gpt4_prompts_file, read_jsonl, write_jsonl, gpt4_gen, read_json, get_comparison_out_file_path, parse_preference_score, parse_score_double



ref_answer_keys = {
    # VICUNA: 'question_ar' if self.lang == AR else 'question',
    # SELF_INSTRUCT: 'question_ar' if self.lang == AR else 'question',
    # SUMMARIZATION: 'question_ar' if self.lang == AR else 'question',
    # SAFETY: "question",
    IQEVAL: 'answerkey',
    FRESHQA: 'answer_0',
    ITC: 'output',
    QUANT: 'gold_answer',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", help="Model_1 name", )
    parser.add_argument("--model2", help="Model_2 name", )
    parser.add_argument("--model1_response", type=str, help="response filepath for model1", )
    parser.add_argument("--model2_response", type=str, help="response filepath for model2", )
    parser.add_argument("--gpt4_prompt_type", type=str, choices=GPT4_PROMPT_TYPE,
                        help="values can be generic, safety, summary", )
    parser.add_argument("--task", type=str, choices=TASKS, help="values can be vicuna, seeds, safety, summary", )
    parser.add_argument("--lang", type=str, choices=[AR, EN], help="language en or ar", default="ar")
    parser.add_argument("--cross", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    prompts = read_json(gpt4_prompts_file)

    if args.cross:
        comparison_out_jsonl_path,comparison_out_xls_path = get_comparison_out_file_path(args.model1,args.model2,args.gpt4_prompt_type,args.task,"cross")

        model1_response_path = args.model1_response if args.model1_response else get_response_out_file_path(args.model1,
                                                                                                       args.task,
                                                                                                       "cross")
        model2_response_path = args.model2_response if args.model2_response else get_response_out_file_path(args.model2,
                                                                                                       args.task,
                                                                                                       "cross")
    else:
        comparison_out_jsonl_path, comparison_out_xls_path = get_comparison_out_file_path(args.model1, args.model2,
                                                                                          args.gpt4_prompt_type,
                                                                                          args.task, args.lang)

        model1_response_path = args.model1_response if args.model1_response else get_response_out_file_path(args.model1,
                                                                                                           args.task,
                                                                                                           args.lang)
        model2_response_path = args.model2_response if args.model2_response else get_response_out_file_path(args.model2,
                                                                                                           args.task,
                                                                                                           args.lang)

    data1 = read_jsonl(model1_response_path)
    data2 = read_jsonl(model2_response_path)

    assert len(data1) == len(
        data2), f"length of comparison data not same \n {args.model1_response} contains {len(data1)} " \
                f"\n {args.model2_response} contains {len(data2)} "

    if os.path.exists(comparison_out_jsonl_path):  # RERUN: fix GPT-4 no response Errors
        with open(comparison_out_jsonl_path) as f:
            old_data = [json.loads(line) for line in f]
    else:
        old_data = [dict()] * len(data1)

    output = []
    for r1, r2, r3 in tqdm(zip(data1, data2, old_data)):
        # GPT-4 evaluation already there  AND scores can be extracted
        if ('GPT4-review' in r3 and r3['GPT4-review'] != 'error') and (
                r3[f'{args.model1}_score'] != -1 and r3[f'{args.model2}_score'] != -1):
            output.append(r3)
            continue

        # ref_answer_key = ref_answer_keys[args.task]

        # Get GPT-4 Evaluation
        gpt4_eval_prompt = prompts[args.lang][args.gpt4_prompt_type]["prompt_template"].format(
            question=r1['question'],
            answer_1=r1['response'],
            answer_2=r2['response'],
            prompt_end=prompts[args.lang][args.gpt4_prompt_type]["prompt_end"],
        )
        review = gpt4_gen(prompts[args.lang][args.gpt4_prompt_type]["system_prompt"], gpt4_eval_prompt, 2048)
        if args.gpt4_prompt_type == "preference":
            scores = parse_preference_score(review)
            r3 = {
                'question': r1["question"],
                args.model1: r1["response"],
                args.model2: r2["response"],
                'GPT4-review': review,
                f'helpful_score': scores[0],
                f'style_score': scores[1],
                f'safety_score': scores[2],
            }
        else:
            scores = parse_score_double(review)

            r3 = {
                'question': r1["question"],
                args.model1: r1["response"],
                args.model2: r2["response"],
                'GPT4-review': review,
                f'{args.model1}_score': scores[0],
                f'{args.model2}_score': scores[1],
            }
        output.append(r3)

    # dump as jsonl
    print(f"Writing to jsonl file at: {comparison_out_jsonl_path}")
    write_jsonl(output, comparison_out_jsonl_path)

    # dump as excel
    try:
        df = pd.DataFrame(output)
        df.to_excel(comparison_out_xls_path, index=False)
    except:
        print(f"Error while creating xls file but jsonl output is already created: {comparison_out_jsonl_path}")


if __name__ == "__main__":
    main()
