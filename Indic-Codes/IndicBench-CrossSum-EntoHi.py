from transformers import AutoTokenizer, AutoModelForCausalLM, logging   # type: ignore
from datasets import load_dataset
from sacrebleu.metrics import CHRF   # type: ignore
import argparse
import torch   # type: ignore
from torch.cuda.amp import autocast  # type: ignore
from torch.nn import DataParallel   # type: ignore

# Step 1: Set up argument parsing
parser = argparse.ArgumentParser(description="Run evaluation on the LLaMA3 model.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# Transformers logging to suppress messages
logging.set_verbosity_error()

# Step 2: Load the LLaMA3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).half().to('cuda')  # Convert model to half precision

# Use DataParallel to leverage multiple GPUs
model = DataParallel(model)

# Set the padding token to the end-of-sequence token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model Loaded")

# Step 3: Load the GoogleIndicGenBench_crosssum_in dataset (test split)
dataset = load_dataset("Cognitive-Lab/GoogleIndicGenBench_crosssum_in", name="hi", split="test")
print("Dataset Loaded")

# Step 4: Define the CHRF metric
chrf = CHRF()

# Initialize variable to keep track of the cumulative CHRF score
total_chrf_score = 0
num_samples = len(dataset)

# Track how many samples we have printed for cross-summarization
printed_summaries = 0

# Step 5: Evaluate the model on the dataset
for id, example in enumerate(dataset):
    
    input_text = example['text']  # Adjust field name if necessary
    target_text = example['summary']  # Adjust field name if necessary
    
    # Tokenize input with truncation and padding (no .to('cuda'))
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Generate output with no_grad and mixed precision for memory optimization
    with torch.no_grad():
        with autocast():
            outputs = model.module.generate(**inputs, max_new_tokens=50)  # Reduced max_new_tokens
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the first two cross-summarizations
    if printed_summaries < 2:
        print(f"Original Text {id + 1}: {input_text}")
        print(f"Generated Summary {id + 1}: {generated_text}")
        print(f"Target Summary {id + 1}: {target_text}\n")
        printed_summaries += 1

    # Compute CHRF score
    chrf_score = chrf.sentence_score(generated_text, [target_text]).score
    
    # Accumulate CHRF score
    total_chrf_score += chrf_score
    
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

# Compute average CHRF score
avg_chrf_score = total_chrf_score / num_samples

# Display the result
print(f"Average CHRF Score: {avg_chrf_score}")
