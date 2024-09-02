import re
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from sklearn.metrics import f1_score
import torch
import gc
from transformers import BitsAndBytesConfig
import argparse

# Transformers logging to suppress message
logging.set_verbosity_error()

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Use quantization to reduce memory footprint
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

# Load the model with quantization directly (no need to use .to() afterwards)
model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

print("Model Loaded")

# Step 2: Load the XQUAD dataset
dataset = load_dataset("Cognitive-Lab/GoogleIndicGenBench_xquad_in", name="hi", split="test")
print("Dataset Loaded")

# Step 3: Preprocess the data
def preprocess_function(examples):
    inputs = ["हिंदी में संदर्भ: " + context + "\n" + "हिंदी में प्रश्न: " + question + "\n" + "हिंदी में उत्तर: " for context, question in zip(examples['context'], examples['question'])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

def extract_answer_from_generated(generated_text, reference_text):
    # Check if the reference text is a substring of the generated text
    if reference_text in generated_text:
        return reference_text
    else:
        # Fall back to token-level comparison
        return generated_text

# Apply preprocessing to the entire dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "title", "lang", "id"])

# Convert dataset to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'answers'])

# Step 4: Run evaluation on the entire dataset
def evaluate_dataset(model, dataset):
    model.eval()
    all_predictions, all_references = [], []

    for i, batch in enumerate(dataset):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        inputs = batch['input_ids'].unsqueeze(0)  # Ensure input is 2D
        attention_mask = batch['attention_mask'].unsqueeze(0)  # Ensure attention_mask is 2D

        inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs  # Move inputs to GPU if available
        attention_mask = attention_mask.to('cuda') if torch.cuda.is_available() else attention_mask

        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                attention_mask=attention_mask, 
                max_new_tokens=50,  # Reduced to 50
                num_beams=5,        # Use beam search for better quality generation
                early_stopping=True
            )

        # Extract only the new tokens generated by the model
        generated_tokens = outputs[:, inputs.shape[-1]:]  # Skip the input tokens
        preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Get reference text
        reference_text = batch['answers'][0]['text']

        # Extract the answer from the generated text
        extracted_answer = extract_answer_from_generated(preds[0], reference_text)

        # Tokenize both for F1 calculation
        reference_tokens = tokenizer(reference_text, return_tensors="pt", truncation=True, padding=True).input_ids[0].tolist()
        generated_tokens = tokenizer(extracted_answer, return_tensors="pt", truncation=True, padding=True).input_ids[0].tolist()

        # Match the length of the predictions and references
        min_length = min(len(reference_tokens), len(generated_tokens))
        matched_reference_tokens = reference_tokens[:min_length]
        matched_generated_tokens = generated_tokens[:min_length]

        all_predictions.extend(matched_generated_tokens)
        all_references.extend(matched_reference_tokens)

        # Debugging: Print out the generated text and reference text for inspection
        if i < 3:  # Adjust this to print more samples if needed
            print(f"Sample {i+1}:")
            print(f"Generated Text: {extracted_answer}")
            print(f"Reference Text: {reference_text}")
            print(f"Predicted Tokens: {matched_generated_tokens}")
            print(f"Reference Tokens: {matched_reference_tokens}")
            print("-" * 100)

        # Delete unnecessary tensors to free up memory
        del inputs, attention_mask, outputs, generated_tokens
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate overall F1 using sklearn for token-level comparison
    f1 = f1_score(all_references, all_predictions, average='weighted', zero_division=1)

    return f1

# Run evaluation on the entire dataset
f1 = evaluate_dataset(model, tokenized_dataset)

# Final output
print(f"Token-Level F1 Score: {f1:.4f}")