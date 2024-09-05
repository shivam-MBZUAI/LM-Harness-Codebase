from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from datasets import load_dataset
from sklearn.metrics import f1_score
import torch

# Transformers logging to suppress message
logging.set_verbosity_error()

# Step 1: Load the LLaMA3 model and tokenizer
model_name = "CohereForAI/aya-23-35B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model on CPU to avoid GPU memory issues
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", offload_folder="offload")

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

print("Model Loaded")

# Step 2: Load the Indic-Bench dataset
dataset = load_dataset("Cognitive-Lab/GoogleIndicGenBench_xorqa_in", name="hi", split="test")
print("Dataset Loaded")

# Step 3: Preprocess the data
def preprocess_function(examples):
    inputs = ["Context: " + context + "\n" + "हिंदी में प्रश्न: " + question + "\n" + "उत्तर केवल हिंदी में दें:" for context, question in zip(examples['context'], examples['question'])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=1024)

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
        inputs = batch['input_ids'].unsqueeze(0)
        attention_mask = batch['attention_mask'].unsqueeze(0)
        
        # Move inputs to GPU and process them in chunks
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = attention_mask.to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                attention_mask=attention_mask, 
                max_new_tokens=50,  # Keep this low to reduce memory usage
                num_beams=3,        # Reduce beams to lower memory consumption
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

        # Clear the CUDA cache after each batch to free up memory
        torch.cuda.empty_cache()
    
    f1 = f1_score(all_references, all_predictions, average='weighted', zero_division=1)
    
    return f1

# Run evaluation on the entire dataset
f1 = evaluate_dataset(model, tokenized_dataset)

# Final output
print(f"Token-Level F1 Score: {f1:.4f}")
