import json

"""

from datasets import load_dataset
dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-Instruct-evals", name="Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_hi__details")
 
data_list = [dict(item) for item in dataset['latest']]
with open('multilingual_mmlu_hi_details.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

"""

# Function to convert format1 entry to format2 entry
def convert_format1_to_format2(entry):
    format2_entry = {
        "ctx_a": entry["translated_ctx_a"],
        "ctx_b": entry["ctx_b"],
        "ctx": entry["translated_ctx_a"] + " " + entry["ctx_b"],
        "endings": entry["translated_endings"],
        "id": entry["source_id"],
        "ind": entry["ind"],
        "activity_label": entry["activity_label"],
        "source_id": entry["source_id"],
        "split": entry["split"],
        "split_type": entry["split_type"],
        "label": entry["label"]
    }
    return format2_entry

# Load JSON files
with open('IndicEval_hellaswag_hi_val.json', 'r', encoding='utf-8') as f1:
    format1_data = json.load(f1)

# Filter and convert format1 entries to format2 entries
format2_data = [convert_format1_to_format2(entry) for entry in format1_data]

# Save the converted entries to a new JSON file
with open('ho_validation.json', 'w', encoding='utf-8') as f2:
    json.dump(format2_data, f2, ensure_ascii=False, indent=4)

print("Conversion complete. The converted entries are saved in 'format2_converted.json'.")

