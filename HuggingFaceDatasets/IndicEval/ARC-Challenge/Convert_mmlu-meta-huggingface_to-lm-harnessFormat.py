import json

"""

from datasets import load_dataset
dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", name="Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_hi__details")
 
data_list = [dict(item) for item in dataset['latest']]
with open('multilingual_mmlu_hi_details.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

"""
def convert_answer_key(answer_key):
    numeric_to_alpha_mapping = {
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D"
    }
    alpha_to_alpha_mapping = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D"
    }
    if answer_key in numeric_to_alpha_mapping:
        return numeric_to_alpha_mapping[answer_key]
    elif answer_key in alpha_to_alpha_mapping:
        return alpha_to_alpha_mapping[answer_key]
    return answer_key


def convert_format1_to_format2(format1_sample):
    choices_text = format1_sample["translated_choices"]["text"]
    answer_key = convert_answer_key(format1_sample["answerKey"])
    format2_sample = {
        "id": "ARC-Challenge/validation/" + format1_sample["id"],
        "answer": answer_key,
        "instruction": format1_sample["translated_question"],
        "option_a": choices_text[0] if len(choices_text) > 0 else "",
        "option_b": choices_text[1] if len(choices_text) > 1 else "",
        "option_c": choices_text[2] if len(choices_text) > 2 else "",
        "option_d": choices_text[3] if len(choices_text) > 3 else ""
    }
    return format2_sample

def convert_all_samples(format1_file, format2_file):
    with open(format1_file, 'r', encoding='utf-8') as infile:
        format1_data = json.load(infile)

    format2_data = [convert_format1_to_format2(sample) for sample in format1_data]

    with open(format2_file, 'w', encoding='utf-8') as outfile:
        json.dump(format2_data, outfile, ensure_ascii=False, indent=4)

# Replace 'input_format1.json' with the path to your input file and 'output_format2.json' with the desired output file path
convert_all_samples('IndicEval_ARC-Challenge_hi_val.json', 'hq_validation.json')

print("Done!!")



