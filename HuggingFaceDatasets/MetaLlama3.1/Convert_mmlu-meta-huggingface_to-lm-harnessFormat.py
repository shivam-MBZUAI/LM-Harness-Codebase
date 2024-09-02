import json

"""

from datasets import load_dataset
dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", name="Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_hi__details")
 
data_list = [dict(item) for item in dataset['latest']]
with open('multilingual_mmlu_hi_details.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

"""

def convert_format1_to_format2(entry, number):
    subtask_name_with_slash = entry["subtask_name"].replace('.', '/')
    id_value = f"{subtask_name_with_slash}/{number}"
    return {
        "instruction": entry["input_question"],
        "option_a": entry["input_choice_list"]["A"],
        "option_b": entry["input_choice_list"]["B"],
        "option_c": entry["input_choice_list"]["C"],
        "option_d": entry["input_choice_list"]["D"],
        "answer": entry["input_correct_responses"][0],
        "id": id_value
    }

def main():
    # Load the format1 JSON file
    with open('Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_hi__details.json', 'r', encoding='utf-8') as f:
        format1_data = json.load(f)
    
    # Convert each entry from format1 to format2 with an incremental number for the id
    format2_data = [convert_format1_to_format2(entry, i+1) for i, entry in enumerate(format1_data)]

    # Write the converted data to a new JSON file
    with open('hm_test.json', 'w', encoding='utf-8') as f: # using hm for hi to differentiate original hi_test.json from llama3.1 mmlu-hi file
        json.dump(format2_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
