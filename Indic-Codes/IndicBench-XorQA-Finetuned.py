from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from sklearn.metrics import f1_score
import torch
import argparse

# Transformers logging to suppress message
logging.set_verbosity_error()

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Step 2: Load the custom LLaMA3 model and tokenizer with mixed precision
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

print("Model Loaded")

# Move the model to cuda:0 explicitly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load the Indic-Bench dataset
dataset = load_dataset("Cognitive-Lab/GoogleIndicGenBench_xorqa_in", name="hi", split="test")
print("Dataset Loaded")

# Step 3: Preprocess the data
def preprocess_function(examples):
    inputs = ["Context: " + context + "\n" + "हिंदी में प्रश्न: " + question + "\n" + "उत्तर केवल हिंदी में दें:" for context, question in zip(examples['context'], examples['question'])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=1024)  # Reduce max_length to 1024

def extract_answer_from_generated(generated_text, reference_text):
    if reference_text in generated_text:
        return reference_text
    else:
        return generated_text

# Apply preprocessing to the entire dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "title", "lang", "split", "oracle_question", "answers"])

# Convert dataset to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'translated_answers'])

def evaluate_dataset(model, dataset):
    model.eval()
    all_predictions, all_references = [], []
    
    for i, batch in enumerate(dataset):
        inputs = batch['input_ids'].to(device).unsqueeze(0)
        attention_mask = batch['attention_mask'].to(device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                attention_mask=attention_mask, 
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True
            )
        
        generated_tokens = outputs[:, inputs.shape[-1]:]
        preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        reference_text = batch['translated_answers'][0]['text']
        extracted_answer = extract_answer_from_generated(preds[0], reference_text)
        
        reference_tokens = tokenizer(reference_text, return_tensors="pt", truncation=True, padding=True).input_ids[0].tolist()
        generated_tokens = tokenizer(extracted_answer, return_tensors="pt", truncation=True, padding=True).input_ids[0].tolist()
        
        min_length = min(len(reference_tokens), len(generated_tokens))
        matched_reference_tokens = reference_tokens[:min_length]
        matched_generated_tokens = generated_tokens[:min_length]
        
        all_predictions.extend(matched_generated_tokens)
        all_references.extend(matched_reference_tokens)
        
        if i < 3:
            print(f"Sample {i+1}:")
            print(f"Generated Text: {extracted_answer}")
            print(f"Reference Text: {reference_text}")
            print(f"Predicted Tokens: {matched_generated_tokens}")
            print(f"Reference Tokens: {matched_reference_tokens}")
            print("-" * 100)
    
    f1 = f1_score(all_references, all_predictions, average='weighted', zero_division=1)
    
    return f1

# Clear CUDA cache before running evaluation
torch.cuda.empty_cache()

# Run evaluation on the entire dataset
f1 = evaluate_dataset(model, tokenized_dataset)

# Final output
print(f"Token-Level F1 Score: {f1:.4f}")
