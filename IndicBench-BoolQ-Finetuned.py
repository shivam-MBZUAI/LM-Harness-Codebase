import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from sklearn.metrics import f1_score
import argparse

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Transformers logging to suppress message
logging.set_verbosity_error()

# Step 1: Load the custom LLaMA3 model and tokenizer with mixed precision
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)  # Use FP16

# Utilize multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Set the padding token to the end-of-sequence token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model Loaded")

# Step 2: Load only 10 examples from the BoolQA dataset (validation split)
dataset = load_dataset("Cognitive-Lab/Indic-BoolQ", name="hi", split="validation")
print("Dataset Loaded")

# Step 3: Initialize variables for F1 score computation
true_labels = []
predicted_labels = []

# Step 4: Evaluate the model on the 10 examples using logits comparison
for id, example in enumerate(dataset):
    target_label = 1 if example['answer'] else 0
    
    # Adding an explicit prompt to guide the model
    input_text_with_prompt = (
        "हिंदी में संदर्भ: " + example['translated_passage'] +
        "\nहिंदी में प्रश्न: " + example['translated_question'] +
        "\nउत्तर: true or false:"
    )

    # Tokenize input
    inputs = tokenizer(input_text_with_prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    # Get logits directly from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[:, -1, :].cpu()  # Focus on the last token generated and move logits to CPU
    true_id = tokenizer.convert_tokens_to_ids("true")
    false_id = tokenizer.convert_tokens_to_ids("false")

    # Compare logits for "true" and "false"
    true_score = logits[:, true_id].item()
    false_score = logits[:, false_id].item()

    # Predict the label based on the higher score
    predicted_label = 1 if true_score > false_score else 0

    if id < 3:
        print(f"Logits for Example {id} - True: {true_score}, False: {false_score}, Predicted: {'true' if predicted_label else 'false'} , Answer: {example['answer']}")
    
    true_labels.append(target_label)
    predicted_labels.append(predicted_label)

# Compute F1 score with zero_division handling
f1 = f1_score(true_labels, predicted_labels, zero_division=1)

# Display the F1 score
print(f"F1 Score: {f1}")
print("Evaluation Complete")
