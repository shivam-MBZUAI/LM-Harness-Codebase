import json

"""

from datasets import load_dataset
dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", name="Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_hi__details")
 
data_list = [dict(item) for item in dataset['latest']]
with open('multilingual_mmlu_hi_details.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

"""

# Load JSON data from format1 file
with open('IndicEval_mmlu_hi_val.json', 'r', encoding='utf-8') as f:
    format1_data = json.load(f)

# Initialize a list to hold the converted data
format2_data = []

# Iterate through each entry in the format1 data
for idx, entry in enumerate(format1_data):
    # Create a new dictionary for format2
    format2_entry = {
        "instruction": entry["translated_question"],
        "option_a": entry["translated_choices"][0],
        "option_b": entry["translated_choices"][1],
        "option_c": entry["translated_choices"][2],
        "option_d": entry["translated_choices"][3],
        "answer": chr(65 + entry["answer"]),  # Convert numeric answer to letter (0 -> A, 1 -> B, etc.)
        "id": f"{entry['subject']}/test/{idx + 1}"  # Create a unique id
    }
    # Add the converted entry to the list
    format2_data.append(format2_entry)

# Save the converted data to a new JSON file
with open('hn_val.json', 'w', encoding='utf-8') as f:
    json.dump(format2_data, f, ensure_ascii=False, indent=4)

print("Conversion complete! The format2.json file has been created.")


